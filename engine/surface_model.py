"""Lift visible rectangular outlines into camera coordinates (+x right, +y down, +z forward)."""
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from engine.geometry import UNIT, transform
from engine.planarity_entry import METHOD as PLANE_RECOVERY


DEFAULTS = Path(__file__).with_name('surface_defaults.json')
SINGLE_WALL_DEPTH = {'wall-recovery-single-v2':'depth-anything-v2','wall-recovery-single-pro':'depth-pro'}
CROP_SPLIT = 'plane-recovery-crop-splits'
PLANE_METHODS = (PLANE_RECOVERY, CROP_SPLIT)
RECOVERY_METHODS = ('wall-recovery', *SINGLE_WALL_DEPTH, *PLANE_METHODS)
BOUNDARY_METHODS = ('regions-first','color-edges','depth-split','wall-depth-anything','strip-mlsd',
                    'wall-and-underside', *RECOVERY_METHODS)
SETTINGS = dict(assumed_vertical_fov_degrees=70, shared_edge_tolerance_px=5,
                minimum_shared_edge_px=10, max_relative_edge_disagreement=.20,
                max_pitch_anisotropy=1.25, min_fold_degrees=12)


def intrinsic(size, focal):
    w, h = size
    return np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.]])


def lens(quads, size):
    """Perpendicular directions supply a focal length for centered square pixels."""
    w, h = size
    centered = intrinsic(size, 1)
    estimates = []
    for q in quads:
        try:
            a, b = (np.linalg.solve(centered, transform(q))[:, :2]).T
        except np.linalg.LinAlgError:
            continue
        if abs(a[2] * b[2]) < 1e-10:
            continue
        squared = -a[:2] @ b[:2] / (a[2] * b[2])
        if (.35*h)**2 < squared < (3*h)**2:
            estimates.append(float(np.sqrt(squared)))
    if estimates:
        return intrinsic(size, np.median(estimates)), 'orthogonal-outline-vanishing-points'
    f = h / (2 * np.tan(np.radians(SETTINGS['assumed_vertical_fov_degrees']) / 2))
    return intrinsic(size, f), 'assumed-70-degree-vertical-fov'


def pitch(image, quad, version, method):
    if method == 'repeat':
        from engine.surface_pitch import fit
        return fit(image, quad, version)
    if method != 'texel':
        raise ValueError(f'Unknown pitch method {method}')
    from engine.texel_grid import texel_grid
    counts = list(map(int, texel_grid(np.asarray(image), quad)))
    grid = dict(phase_blocks=[0, 0], grid_to_image=(transform(quad) @
                np.diag([1/counts[0], 1/counts[1], 1])).tolist())
    return dict(counts=counts, grid=grid, confidence=None,
                reason='Legacy texel spacing; first chart corner assumed to lie on a seam.')


def canonical(quad):
    """Orient the two chart coordinates predominantly right and down in the image."""
    q = np.asarray(quad, float)
    choices = [np.roll(a, n, axis=0) for a in (q, q[::-1]) for n in range(4)]
    def alignment(p):
        u, v = p[1] + p[2] - p[0] - p[3], p[3] + p[2] - p[0] - p[1]
        return u[0]/max(1e-9, np.linalg.norm(u)) + v[1]/max(1e-9, np.linalg.norm(v))
    return max(choices, key=alignment).tolist()


def lift(q, K, result):
    try:
        ray = np.linalg.solve(K, transform(q))
    except np.linalg.LinAlgError:
        return None
    xyz = np.column_stack((UNIT, np.ones(4))) @ ray.T
    if np.min(xyz[:, 2]) <= .01:
        return None
    normal = np.cross(ray[:, 0], ray[:, 1])
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return None
    normal *= -np.sign(normal @ xyz[0]) / norm
    counts = result.get('counts')
    scale = None
    if counts is not None:
        scales = counts / np.linalg.norm(ray[:, :2], axis=0)
        if np.isfinite(scales).all() and min(scales) > 0 and max(scales)/min(scales) <= SETTINGS['max_pitch_anisotropy']:
            scale = float(np.mean(scales) * xyz[:, 2].mean())
    return xyz / xyz[:, 2].mean(), normal, scale


def shared_edge(one, two):
    """Only stitch two complete, nearly coincident edges of differently oriented planes."""
    q, xyz, normal = one
    r, pqr, other = two
    if abs(normal @ other) > np.cos(np.radians(SETTINGS['min_fold_degrees'])):
        return None
    best = None
    for a in range(4):
        ai = [a, (a+1)%4]
        if np.linalg.norm(q[ai[1]] - q[a]) < SETTINGS['minimum_shared_edge_px']:
            continue
        for b in range(4):
            for bi in ([b, (b+1)%4], [(b+1)%4, b]):
                miss = np.linalg.norm(q[ai] - r[bi], axis=1).max()
                if miss > SETTINGS['shared_edge_tolerance_px'] or (best and miss >= best[0]):
                    continue
                ratios = xyz[ai, 2] / pqr[bi, 2]
                if max(ratios)/min(ratios) <= 1 + SETTINGS['max_relative_edge_disagreement']:
                    best = float(miss), float(np.log(ratios).mean()), ai, bi
    return best


def assemble(proposals, pitches, size, diagnostic_K=None):
    """Solve connected components' relative depth from coincident edge endpoints."""
    if diagnostic_K is None:
        K, source = lens([p['quad'] for p in proposals], size)
    else:
        K, source = diagnostic_K, 'recorded-camera-diagnostic'
    planes, items = [], {}
    for p, result in zip(proposals, pitches, strict=True):
        row = dict(p, pitch=result)
        lifted = lift(p['quad'], K, result)
        if lifted:
            xyz, normal, scale = lifted
            i = len(planes)
            items[i] = np.asarray(p['quad']), xyz, normal, scale
            row.update(normal_camera=normal.tolist())
        planes.append(row)
    edges, graph = [], {i: [] for i in items}
    for i in items:
        for j in items:
            if j <= i:
                continue
            match = shared_edge(items[i][:3], items[j][:3])
            if match is None:
                continue
            miss, log_ratio, ai, bi = match
            direct_i, direct_j = items[i][3], items[j][3]
            if direct_i and direct_j and abs(np.log(direct_j/direct_i)-log_ratio) > np.log(1.25):
                continue
            graph[i].append(j)
            graph[j].append(i)
            edges.append((i, j, log_ratio, dict(planes=[planes[i]['id'], planes[j]['id']],
                         corners=[ai, bi], endpoint_max_px=miss)))
    todo = set(items)
    component = 0
    while todo:
        root = min(todo)
        stack, group = [root], []
        while stack:
            i = stack.pop()
            if i in todo:
                todo.remove(i)
                group.append(i)
                stack.extend(graph[i])
        index = {i: k for k, i in enumerate(group)}
        equations, target = [], []
        def constraint(weights, value):
            row = np.zeros(len(group))
            for i, weight in weights.items():
                row[index[i]] = weight
            equations.append(row)
            target.append(value)
        for i, j, ratio, _ in edges:
            if i in index:
                constraint({i: -1, j: 1}, ratio)
        anchors = [i for i in group if items[i][3] is not None]
        for i in anchors:
            constraint({i: 1}, np.log(items[i][3]))
        if not anchors:
            constraint({root: 1}, 0.)
        scales = np.exp(np.linalg.lstsq(equations, target, rcond=None)[0])
        for i in group:
            planes[i].update(component=component, units='blocks' if anchors else 'relative',
                             scale_source=('local-pitch' if i in anchors else 'shared-edge') if anchors else 'unknown',
                             corners_camera=(items[i][1] * scales[index[i]]).tolist())
        component += 1
    return dict(lens_source=source, intrinsics=K.tolist(), planes=planes,
                shared_edges=[edge[-1] for edge in edges])


def proposals_for(image, version, method, normals=None, depth=None):
    if method == 'rectangle-finder':
        method = 'normal-regions'
    if method == 'mask-first':
        from engine.mask_first import regions
        from engine.surface_fields import infer
        if normals is None:
            normals = infer(image, 'normal')
        if depth is None:
            depth = infer(image, 'depth')
        return regions(image, normals, depth, version)
    if method == 'contact':
        from engine.camera_grid import predict
        parts = predict(image, version)['planes']
        return [dict(id=f'contact-{i}', quad=canonical(p['quad'])) for i,p in enumerate(parts)]
    if method in ('normal-rank','normal-regions'):
        if normals is None:
            from engine.surface_fields import infer
            normals=infer(image,'normal')
        if method=='normal-rank':
            from engine.normal_rank import detect
        else:
            from engine.normal_regions import detect
        parts=detect(image,normals,version=version)
        from engine.surface_fields import observed_normal
        for p in parts:
            p['observed_normal_camera']=observed_normal(p['quad'],normals,image.size)
    elif method in ('edges','edges+shade'):
        from engine.surface_edges import detect_planes
        parts=detect_planes(image,version=version,shading=method=='edges+shade')
    else:
        raise ValueError(f'Unknown outline method {method}')
    reordered = {}
    for p in parts:
        old = p['quad']
        p['quad'] = canonical(old)
        reordered[p['id']] = [p['quad'].index(corner) for corner in old]
    for p in parts:
        for link in p.get('corner_junctions',[]):
            link['corner'] = reordered[p['id']][link['corner']]
            link['neighbor_corner'] = reordered[link['neighbor']][link['neighbor_corner']]
    return parts


def _count_regions(image, planes, version, method, atlas, out, shape_matcher=None, shadow_field=None):
    from engine.surface_sample import sample_surface
    from engine.surface_dimensions import dimensions
    from engine import texture_guided_count

    if method not in ('legacy-repeat', 'masked-repeat', 'atlas-repeat', texture_guided_count.METHOD):
        raise ValueError(f'Unknown Block Counter method {method}')
    measurements = []
    faces = []
    phase_report = None
    if method in ('masked-repeat', texture_guided_count.METHOD) and atlas is not None:
        from engine.atlas_phase import register_phase
        phase_report = dict(status='enabled', offsets_added=0, surface_errors=0)
    samples_dir = out / 'samples'
    samples_dir.mkdir(exist_ok=True)
    for i, plane in enumerate(planes):
        sample = sample_surface(image, plane)
        row = dict(id=plane['id'], method=method)
        if sample is None:
            if method == texture_guided_count.METHOD:
                row['method'] = 'masked-repeat'
            row.update(counts=[None, None], phase=[None, None], reason='No usable sampling chart',
                       dimensions=dict(spans=[None, None], blocks=[None, None]))
        else:
            if method == 'atlas-repeat':
                from engine.atlas_count import count
                result = count(sample, atlas, version=version)
            elif method == texture_guided_count.METHOD:
                result, sample = texture_guided_count.count_surface(image, plane, sample, version=version)
            elif method == 'legacy-repeat':
                from engine.legacy_count import count
                result = count(sample, version=version)
            else:
                from engine.masked_count import count
                result = count(sample, version=version)
            rectified = Image.fromarray(np.asarray(sample['rgb'], dtype=np.uint8))
            rectified.putalpha(Image.fromarray(np.asarray(sample['mask'], dtype=np.uint8)*255))
            path = samples_dir / f'surface-{i:03}.png'
            rectified.save(path)
            row.update(quad=np.asarray(sample['quad']).tolist(), sample=path.relative_to(out).as_posix())
            if phase_report is not None:
                try:
                    registered = register_phase(sample, result, atlas)
                except Exception as exc:
                    # Only the optional atlas step can fall back. Keep the measured grid.
                    row['atlas_phase_error'] = f'{type(exc).__name__}: {exc}'
                    phase_report['status'] = 'partial'
                    phase_report['surface_errors'] += 1
                else:
                    if registered['phase'] != result['phase']:
                        row['phase_before_atlas'] = result['phase']
                        phase_report['offsets_added'] += sum(
                            old is None and new is not None
                            for old, new in zip(result['phase'], registered['phase']))
                    result = registered
            row.update(result)
            row['dimensions'] = dimensions(plane, sample['quad'], image.size, result['counts'])
            from engine.surface_tiles import complete_footprints
            from engine.face_evidence import effective_width, MIN_FACE_WIDTH
            row['visible_cells'] = []
            row['under_five_px_faces'] = []
            for cell, quad in complete_footprints(sample, result):
                width = effective_width(quad)
                target = 'visible_cells' if width >= MIN_FACE_WIDTH else 'under_five_px_faces'
                record = dict(face_index=len(faces), surface_id=plane['id'], cell=cell,
                              quad=canonical(quad), effective_width_px=width,
                              normal_camera=plane.get('normal_camera'))
                row[target].append(record)
                faces.append(record)
            if not row['visible_cells'] and not row['under_five_px_faces']:
                row['material_status'] = 'No complete cell has a supported position inside this mask.'
        measurements.append(row)
    report = dict(version=version, units='blocks across and down the rectified sampling chart',
                  shape_artifact='surfaces.json', surfaces=measurements)
    if phase_report is not None:
        report['atlas_phase'] = phase_report
    if atlas is not None:
        from engine.atlas_count import rank_tile
        from engine.block_face_groups import assign_block_materials
        def material(tile):
            if shape_matcher is None:
                return rank_tile(tile, atlas, version=version)
            patch = tile['rgb']
            shadow = None
            if shadow_field is not None:
                from engine.blocker_shadow import cell_shadow, corrected
                gains, shadow = cell_shadow(shadow_field, tile['quad'], len(patch))
                patch = corrected(patch, gains)
            if getattr(shape_matcher, 'sample_size', None) == 16:
                from engine.geometry import face
                patch = face(np.asarray(image), tile['quad'], 16)
            result = shape_matcher.rank_patch(patch)
            result['block_id'] = 'minecraft:' + result['block_id']
            result['ranked_ids'] = ['minecraft:' + b for b in result['ranked_ids']]
            result['same_texture_ids'] = sorted('minecraft:' + b for b in shape_matcher.mapping[result['texture']])
            if shadow is not None:
                result['shadow'] = shadow
            return result

        groups = assign_block_materials(image, faces, material)
        report['block_groups'] = groups
        attempted = sum(g['selected_index'] is not None for g in groups)
        report['material_block_counts'] = dict(total=len(groups), attempted=attempted,
                                                unresolved=len(groups)-attempted)
        report['material_face_counts'] = dict(total=len(faces), attempted=attempted,
            inherited=sum(len(g['face_indices'])-1 for g in groups if g['selected_index'] is not None),
            under_five_px=sum(len(row.get('under_five_px_faces', [])) for row in measurements))
    else:
        from engine.block_face_groups import group_faces
        groups = group_faces(faces, image.size)
        for block_index, group in enumerate(groups):
            group['block_index'] = block_index
            for idx in group['face_indices']:
                faces[idx]['block_index'] = block_index
                faces[idx]['material_source_face'] = group['selected_index']
        report['block_groups'] = groups
    return report


def _inset_layout_counts(image, version, original, counts, hosts, counter, atlas):
    """Keep the original global layout and recount just the owners of new holes."""
    with TemporaryDirectory() as tmp:
        old_rows = _count_regions(image, [p for p in original if p['id'] in hosts],
                                  version, counter, atlas, Path(tmp))['surfaces']
    updated = {r['id']:r for r in counts['surfaces']}
    for old in old_rows:
        row = updated[old['id']]
        dims = row['dimensions']
        if None in dims['blocks']:
            for axis, value in enumerate(dims['blocks']):
                if value is not None and old['dimensions']['blocks'][axis] is None:
                    # Cutting a hole cannot independently confirm a new axis
                    # of the host when the other axis is still unmeasured.
                    row.setdefault('unconfirmed_host_dimension', []).append(dict(axis=axis, estimate=value))
                    dims['blocks'][axis] = None
                    dims['spans'][axis] = None
        updated[old['id']] = old
    return dict(counts, surfaces=[updated[p['id']] for p in original])


def _material_assets(version, counter, matcher, explicit_path):
    """Use the requested directory or the first usable installation, in search order."""
    from engine.atlas_count import load_atlas, prepared_atlas_paths

    paths = [explicit_path] if explicit_path is not None else prepared_atlas_paths(version)
    required = counter == 'atlas-repeat'
    if required and not paths:
        paths = [None]  # The explicit atlas-based counter also supports local generation.
    failures = []
    for index, path in enumerate(paths):
        try:
            atlas = load_atlas(path, version=version, verify_material_index=matcher in ('shape', 'shape-shadow', 'exposure'))
            shape_matcher = None
            if matcher in ('shape', 'shape-shadow'):
                from engine.blocker_shape import ShapeMatcher
                shape_matcher = ShapeMatcher(path, catalog=atlas.material_index)
            return atlas, shape_matcher, path, failures
        except Exception as exc:
            if required and index == len(paths) - 1:
                raise
            failures.append(dict(path=str(path), reason=f'{type(exc).__name__}: {exc}'))
    return None, None, None, failures


def build(image_path, version, out, *, outlines=None, order_method=None, through='surfaces',
          counter=None, atlas_path=None, material_match=None, context_px=None, boundary=None, library=None,
          joint_foreground=True):
    camera_atlas_path = atlas_path
    metric_camera = None
    inset_events = []
    phase_events = []
    count_confirmed_outlines = []
    pre_inset = None
    eave_events = []
    smooth_outline = {}
    common_wall_regions = []
    scene_context = None
    exterior_only = None
    exterior_events = []
    from engine.atlas import digest
    from engine.surface_fields import MODELS, SETTINGS as inference, infer
    if version != '26.3':
        raise ValueError('The surface methods support Java 26.3.')
    if through not in ('surfaces', 'counts', 'placement'):
        raise ValueError(f'Unknown surface stage {through}')
    defaults = json.loads(DEFAULTS.read_text())
    outlines = outlines or defaults['outlines']
    mask_first = outlines == 'mask-first'
    if through == 'placement' and not mask_first:
        raise ValueError('Visible placement requires Regions First surface masks.')
    if boundary is not None and (not mask_first or boundary not in BOUNDARY_METHODS):
        raise ValueError('Select a Regions First boundary method for a mask-first surface build.')
    if context_px is not None and (not mask_first or not isinstance(context_px, int) or context_px < 0):
        raise ValueError('Surface context requires Regions First and a nonnegative whole number of pixels.')
    if order_method is None and not mask_first:
        order_method = defaults['order']
    with Image.open(image_path) as src:
        image = Image.fromarray(np.asarray(src.convert('RGB')))
    fields = {}
    if mask_first or outlines.startswith('normal-') or outlines == 'rectangle-finder':
        fields['normal'] = infer(image, 'normal')
    if mask_first or order_method == 'depth-anything':
        fields['depth'] = infer(image, 'depth')
    patches = proposals_for(image, version, outlines, fields.get('normal'), fields.get('depth'))
    methods = dict(outlines=outlines)
    if mask_first:
        from engine.surface_perimeter import DEFAULT_CONTEXT_PX, context_crop, masked_crop, with_outline
        boundary = boundary or defaults['boundary']
        boundary_settings = {}
        if boundary in ('color-edges','wall-depth-anything','strip-mlsd','wall-and-underside',*RECOVERY_METHODS):
            from engine.surface_boundary_refine import SETTINGS as contrast_settings, refine
            patches=refine(image,patches,version)
            boundary_settings = contrast_settings
            if boundary != 'color-edges':
                boundary_settings = dict(contrast_alignment=contrast_settings)
                methods['initial_boundary'] = 'color-edges'
            if boundary in ('wall-depth-anything','wall-and-underside',*RECOVERY_METHODS):
                from engine.normal_regions import SETTINGS as rectangle_settings
                from engine.surface_depth_split import SETTINGS as vote_settings
                from engine.surface_wall_split import SETTINGS as split_settings, refine as divide_walls
                rectangles = proposals_for(image, version, 'rectangle-finder', fields['normal'])
                # The original rectangle finder's normals are the chart normals after camera lifting.
                # Computing these orientations needs neither a texture pitch nor a block scale.
                camera, _ = lens([r['quad'] for r in rectangles], image.size)
                for rect in rectangles:
                    orientation = lift(rect['quad'], camera, {})
                    if orientation is not None:
                        rect['normal_camera'] = orientation[1].tolist()
                patches = divide_walls(image, patches, fields['depth'], rectangles)
                methods.update(split_hints='rectangle-finder', split_depth='depth-anything-v2')
                boundary_settings.update(rectangle_finder=rectangle_settings, rectangle_lift=SETTINGS,
                                         depth_vote=vote_settings, wall_split=split_settings)
            if boundary in RECOVERY_METHODS:
                from engine.recovery_evidence import infer as recovery_evidence
                from engine.surface_junction_split import SETTINGS as junction_settings, refine as split_junctions
                # Prompts come from the original walls. Junctions and undersides are later partitions.
                recovery, mobile_masks = recovery_evidence(image_path, patches)
                patches = split_junctions(image, patches, {'scalelsd': recovery['lines']}, fields['normal'], fields['depth'])
                methods.update(junction_line_detector='ScaleLSD v2', wall_partition_model='MobileSAM TinyViT')
                boundary_settings.update(junction_split=junction_settings, recovery_inference=recovery['provenance'])
            if boundary in ('strip-mlsd','wall-and-underside',*RECOVERY_METHODS):
                from engine import mlsd_tiny
                from engine.surface_strip_split import SETTINGS as strip_settings, refine as split_strips
                evidence = mlsd_tiny.detect(image)
                patches = split_strips(image, patches, {'mlsd':evidence}, fields['normal'], fields['depth'])
                methods['line_detector'] = mlsd_tiny.NAME
                boundary_settings.update(strip_split=strip_settings, line_inference=mlsd_tiny.SETTINGS,
                                         detected_segments=len(evidence['segments']))
            if boundary == 'wall-and-underside':
                methods['boundary_sequence'] = ['color-edges','wall-depth-anything','strip-mlsd']
            if boundary in RECOVERY_METHODS:
                from benchmark.mobile_wall_split import SETTINGS as mobile_settings, refine as mobile_split
                patches, events = mobile_split(image, patches, mobile_masks, recovery['records'], recovery['selected'])
                methods['boundary_sequence'] = ['color-edges','wall-depth-anything','junction-cuts','strip-mlsd','mobile-wall-split']
                boundary_settings.update(mobile_split=mobile_settings, mobile_splits=events)
                if boundary in SINGLE_WALL_DEPTH or boundary in PLANE_METHODS:
                    from engine import single_wall_gate
                    inventory = single_wall_gate.prepare(image,patches,mobile_masks,recovery['records'],recovery['selected'])
                    depth_name = SINGLE_WALL_DEPTH.get(boundary,'depth-anything-v2')
                    if depth_name=='depth-pro':
                        from engine.recovery_evidence import depth_pro
                        vote_field, provenance = depth_pro(image_path,recovery['provenance']['device'])
                        boundary_settings['depth_pro_inference'] = provenance
                    else:
                        vote_field = single_wall_gate.depth_grid(fields['depth'],image.size)
                    patches, decisions = single_wall_gate.apply(image,patches,mobile_masks,recovery['records'],inventory,vote_field)
                    methods['boundary_sequence'].append('single-wall-split')
                    methods['single_wall_depth'] = depth_name
                    boundary_settings.update(single_wall_rule=single_wall_gate.SETTINGS,single_wall_votes=decisions)
                if boundary in PLANE_METHODS:
                    from engine import planarity_entry, planarity_fusion
                    evidence, provenance = planarity_entry.infer(image_path,recovery['provenance']['device'],
                                                                 crops=boundary==CROP_SPLIT)
                    initial_count = len(patches)
                    patches, events = planarity_fusion.predict(image,patches,evidence,planarity_entry.PRESET)
                    methods.update(plane_model='Pixel-wise Planarity',plane_fusion=planarity_entry.PRESET)
                    methods['boundary_sequence'].append(PLANE_RECOVERY)
                    boundary_settings.update(planarity=dict(inference=provenance,rule=planarity_fusion.SETTINGS,
                                             input_surfaces=initial_count,output_surfaces=len(patches),events=events))
                    if boundary == CROP_SPLIT:
                        from engine.crop_splits import predict as split_existing
                        from engine.planarity_crops import registered_tiles
                        before = len(patches)
                        patches, events = split_existing(image, patches, registered_tiles(evidence))
                        methods['boundary_sequence'].append(CROP_SPLIT)
                        boundary_settings['crop_splits'] = dict(input_surfaces=before, output_surfaces=len(patches),
                                                                events=events, inference=provenance['crops'])
        elif boundary=='depth-split':
            from engine.surface_depth_split import SETTINGS as boundary_settings, split
            patches=split(image,patches,fields['depth'],version)
        methods['surface_boundary']=boundary
        if boundary == CROP_SPLIT:
            from engine.sky_filter import filter_surface, sky_masks
            from engine import photo_edges, sky_gap, strip_neighbor
            blue, cloud = sky_masks(image)
            patches, sky_stats = filter_surface(image, patches, masks=(blue, cloud))
            methods['background_filter'] = 'clear-noon-sky'
            methods['boundary_sequence'].append(methods['background_filter'])
            boundary_settings['background_filter'] = sky_stats
            gap = sky_gap.gap_mask(image, blue, cloud)
            patches, gap_stats = filter_surface(image, patches, masks=(gap, np.zeros_like(gap)))
            methods['opening_sky_filter'] = sky_gap.METHOD
            methods['boundary_sequence'].append(sky_gap.METHOD)
            boundary_settings['opening_sky_filter'] = dict(settings=sky_gap.SETTINGS, **gap_stats)
            patches, boundary_settings['strip_cleanup'] = strip_neighbor.clean_surfaces(image, patches)
            methods['strip_cleanup'] = strip_neighbor.METHOD
            methods['boundary_sequence'].append(strip_neighbor.METHOD)
            patches, boundary_settings['photo_edges'] = photo_edges.refine_surfaces(image, patches)
            methods['photo_edges'] = photo_edges.METHOD
            methods['boundary_sequence'].append(photo_edges.METHOD)
            from engine import guarded_wall_outlines, skyline_extent
            patches, events = guarded_wall_outlines.refine(image, patches)
            boundary_settings['diagonal_wall_join'] = dict(settings=guarded_wall_outlines.SETTINGS, events=events)
            methods['diagonal_wall_join'] = guarded_wall_outlines.METHOD
            methods['boundary_sequence'].append(guarded_wall_outlines.METHOD)
            patches, events = skyline_extent.refine(image, patches, blue | cloud)
            boundary_settings['skyline_extent'] = dict(events=events)
            methods['skyline_extent'] = skyline_extent.METHOD
            methods['boundary_sequence'].append(skyline_extent.METHOD)
            from engine import inset_returns
            pre_inset = patches
            patches, inset_events, phase_events, count_confirmed_outlines, metric_camera = inset_returns.detect(
                image, patches, version, atlas_path=camera_atlas_path)
            from engine import eave_pinholes
            patches, eave_events = eave_pinholes.refine(image, patches)
            methods['visible_insets'] = inset_returns.METHOD
            methods['inset_texture'] = inset_returns.TEXTURE_METHOD
            methods['inset_outlines'] = inset_returns.OUTLINE_METHOD
            methods['boundary_sequence'].extend([inset_returns.METHOD, inset_returns.TEXTURE_METHOD, inset_returns.OUTLINE_METHOD])
            methods['eave_outline'] = eave_pinholes.METHOD
            methods['boundary_sequence'].append(eave_pinholes.METHOD)
            boundary_settings['eave_outline'] = dict(events=eave_events)
            boundary_settings['visible_insets'] = dict(events=inset_events, texture_events=phase_events, count_confirmed_outlines=count_confirmed_outlines)
            # Confirm only intact neighboring masks. Existing inset events keep
            # their original hosts and placement/count bookkeeping.
            if not (inset_events or phase_events or count_confirmed_outlines):
                from engine import jagged_texture_join
                patches, joins, metric_camera = jagged_texture_join.refine(
                    image, patches, version, camera=metric_camera, atlas_path=camera_atlas_path)
                pre_inset = patches
                methods['boundary_sequence'].append(jagged_texture_join.METHOD)
                boundary_settings['jagged_texture_join'] = dict(events=joins)
            from engine import smooth_atlas_outline
            def outline_atlas():
                return _material_assets(version, None, 'atlas', camera_atlas_path)[0]
            patches, smooth_outline, common_wall_regions = smooth_atlas_outline.with_common_wall(image, patches, version, atlas_loader=outline_atlas)
            methods['boundary_sequence'].append(smooth_atlas_outline.METHOD)
            boundary_settings['smooth_atlas_outline'] = smooth_outline
            from engine import exterior_foreground
            if metric_camera is None and any(part.get('normal_camera') is not None for part in patches):
                from engine.visible_placement import seam_camera
                metric_camera = seam_camera(image, version, atlas_path=camera_atlas_path)
            revised, exterior_events = exterior_foreground.refine(image, patches, metric_camera)
            if exterior_events:
                scene_context = [with_outline(p) for p in patches]
                patches = revised
                methods['boundary_sequence'].append('exterior-connected-backdrop')
                boundary_settings['exterior_foreground'] = dict(events=exterior_events)
        context_px = DEFAULT_CONTEXT_PX if context_px is None else context_px
        patches = [with_outline(part) for part in patches]
        if boundary == CROP_SPLIT:
            from engine import horizon_continuity
            foreground, horizon_events = horizon_continuity.refine(image, patches)
            if horizon_events:
                # The distant terrain still constrains camera and plane fits.
                # Separate that evidence from the visible structure outlines.
                exterior_only = patches
                if scene_context is None:
                    scene_context = patches
                patches = [part if 'outline' in part else with_outline(part) for part in foreground]
            methods['boundary_sequence'].append('horizon-continuity')
            boundary_settings['horizon_continuity'] = dict(events=horizon_events)
        crop_dir = out/'surface-crops'
        crop_dir.mkdir(parents=True, exist_ok=True)
        for i, part in enumerate(patches):
            crop = masked_crop(image, part)
            if crop is not None:
                photo, box = crop
                path = crop_dir/f'surface-{i:03}.png'
                photo.save(path)
                part['source_crop'] = dict(path=path.relative_to(out).as_posix(), xyxy=list(box))
                context, window, local = context_crop(image, part, context_px, mask_box=box)
                context_path = crop_dir/f'surface-{i:03}-context.png'
                context.save(context_path)
                part['context_crop'] = dict(path=context_path.relative_to(out).as_posix(), xyxy=list(window),
                                            mask_xyxy_in_crop=list(local), requested_margin_px=context_px,
                                            margins_ltrb_px=[local[0],local[1],context.width-local[2],context.height-local[3]])
        from engine.outline_issues import outline_issues
        model = dict(footprint='support', image_size=list(image.size), planes=patches,
                     outline_issues=outline_issues(patches,image.size),
                     outline_coordinates='closed outer and hole rings on support-grid pixel edges',
                     sampling_chart_coordinates='optional four source-image points for rectification')
        if order_method == 'geometry':
            charted = [p for p in patches if 'quad' in p]
            lifted = assemble(charted, [dict(counts=None) for p in charted], image.size)
            by_id = {p['id']: p for p in lifted.pop('planes')}
            model.update(lifted, planes=[by_id.get(p['id'], p) for p in patches])
            methods.update(lens=lifted['lens_source'], lift='shared-edges-v1')
    else:
        pitches = [pitch(image, p['quad'], version, defaults['pitch']) for p in patches]
        model = assemble(patches, pitches, image.size)
        methods.update(pitch=defaults['pitch'], lens=model['lens_source'], lift='shared-edges-v1')
    if order_method is not None:
        from engine.surface_order import order
        methods['order'] = order_method
        field = fields['depth'] if order_method == 'depth-anything' else None
        model['relative_depth_order'] = order(model['planes'], image.size, field, use_masks=mask_first)
    out.mkdir(parents=True,exist_ok=True)
    (out/'surfaces.json').write_text(json.dumps(model, allow_nan=False, indent=2))
    manifest = dict(version=version, through=through, image_sha256=digest(image_path), stages=methods,
                    artifact='surfaces.json', artifacts=dict(surfaces='surfaces.json'))
    if 'lift' in methods:
        manifest['settings'] = SETTINGS
    if fields:
        manifest['model_sha256'] = {k: MODELS[k][2] for k in fields}
        manifest['inference_settings'] = inference
        if 'line_detector' in methods:
            manifest['model_sha256']['mlsd'] = mlsd_tiny.MODEL[2]
        if boundary in RECOVERY_METHODS:
            manifest['model_sha256'].update({name: recovery['provenance'][name]['checkpoint_sha256']
                                             for name in ('scalelsd', 'mobilesam')})
            if methods.get('single_wall_depth')=='depth-pro':
                manifest['model_sha256']['depth_pro']=boundary_settings['depth_pro_inference']['checkpoint_sha256']
            if boundary in PLANE_METHODS:
                manifest['model_sha256']['pixelwise_planarity']=boundary_settings['planarity']['inference']['checkpoint_sha256']
    if mask_first:
        from engine.mask_first import SETTINGS as finder_settings
        manifest['surface_finder'] = dict(name='Surface Finder', method='Regions First',
                                          directions='MoGe-2', relative_depth='Depth Anything V2',
                                          settings=finder_settings, boundary=boundary, boundary_settings=boundary_settings,
                                          outline='exact-support-perimeter')
        manifest['artifacts']['source_crops'] = 'surface-crops/'
        manifest['surface_finder']['context_crop'] = dict(margin_px=context_px, coordinates='source image pixels',
                                                         purpose='context only; support and counting input unchanged')
    elif outlines in ('rectangle-finder', 'normal-regions'):
        from engine.normal_regions import SETTINGS as rectangle_settings
        manifest['surface_finder'] = dict(name='Surface Finder', method='Rectangle Finder',
                                          implementation='normal-regions', directions='MoGe-2',
                                          settings=rectangle_settings, outline='fitted quadrilateral')
    if through in ('counts', 'placement'):
        measured_patches = patches
        if smooth_outline.get('events'):
            measured_patches = smooth_atlas_outline.measurement_parts(measured_patches)
        count_parts = eave_pinholes.measurement_parts(measured_patches) if eave_events else measured_patches
        counter = counter or defaults['block_count']
        atlas = None
        shape_matcher = None
        shadow_field = None
        atlas_error = None
        if atlas_path is not None or material_match or counter == 'atlas-repeat':
            material_match = material_match or 'atlas'
            atlas, shape_matcher, atlas_path, failures = _material_assets(version, counter, material_match, atlas_path)
            if failures:
                manifest['atlas_search_errors'] = failures
            if atlas is None:
                reason = failures[-1]['reason'] if failures else f'No prepared Java {version} texture atlas found'
                atlas_error = dict(status='unavailable', requested_matcher=material_match,
                                   reason=reason, offsets_added=0)
                logging.getLogger(__name__).warning('Using measured grids without material IDs: %s', reason)
                methods['material_matching'] = 'unavailable'
            else:
                manifest['atlas'] = dict(atlas.provenance, directory=str(atlas_path) if atlas_path else None)
                methods['material_matching'] = material_match
        if atlas is not None and material_match == 'shape-shadow':
            from engine.blocker_shadow import surface_shadow_map
            shadow_field = surface_shadow_map(np.asarray(image.convert('RGB')), [p for p in count_parts if p.get('support')])
        if atlas is not None and material_match == 'exposure':
            from engine.block_exposure import ExposureMatcher
            shape_matcher = ExposureMatcher(atlas_path, library, catalog=atlas.material_index)
            manifest['material_library_sha256'] = digest(library)
        methods['block_grouping'] = 'convex-shared-edges'
        counts = _count_regions(image, count_parts, version, counter, atlas, out, shape_matcher, shadow_field)
        placement_parts, placement_counts = count_parts, counts
        if through == 'placement' and scene_context is not None:
            original = (smooth_atlas_outline.measurement_parts(scene_context)
                        if smooth_outline.get('events') else scene_context)
            placement_parts = eave_pinholes.measurement_parts(original) if eave_events else original
            with TemporaryDirectory() as temporary:
                placement_counts = _count_regions(image, placement_parts, version, counter, atlas,
                                                  Path(temporary), shape_matcher, shadow_field)
        if inset_events or count_confirmed_outlines:
            placement_parts = [with_outline(p) for p in pre_inset]
            placement_counts = _inset_layout_counts(image, version, placement_parts, placement_counts,
                {e['host'] for e in inset_events + count_confirmed_outlines}, counter, atlas)
        if inset_events:
            from engine.visible_inset_cells import observed_back_spans
            visible_spans = observed_back_spans(image.size, metric_camera, measured_patches, inset_events, phase_events)
            for row in counts['surfaces']:
                if row['id'] in visible_spans:
                    row['observed_visible_grid_span'] = visible_spans[row['id']]
        if atlas_error is not None:
            counts['atlas_phase'] = atlas_error
        if 'atlas_phase' in counts:
            manifest['atlas_phase'] = counts['atlas_phase']
            if counts['atlas_phase']['status'] == 'partial':
                logging.getLogger(__name__).warning('Used measured offsets on %d surfaces after atlas errors; see counts.json',
                                                    counts['atlas_phase']['surface_errors'])
        (out/'counts.json').write_text(json.dumps(counts, allow_nan=False, indent=2))
        manifest['artifact'] = 'counts.json'
        manifest['artifacts'].update(counts='counts.json', samples='samples/')
        methods['block_count'] = counter
        from engine import texture_guided_count
        if counter == texture_guided_count.METHOD:
            manifest['block_counter'] = dict(primary='masked-repeat', fallback=counter,
                primary_settings=texture_guided_count.MASKED_SETTINGS, fallback_settings=texture_guided_count.SETTINGS,
                envelope_settings=texture_guided_count.ENVELOPE_SETTINGS,
                accepted_fallbacks=sum(r['method']==counter and 'sample' in r for r in counts['surfaces']))
        if through == 'placement':
            from engine.export import JAVA_26_3_FORMAT, write_pack
            from engine.visible_placement import place
            cells, status = place(image, version, placement_parts, counts=placement_counts,
                                  atlas_path=camera_atlas_path, joint_foreground=joint_foreground, camera=metric_camera)
            if inset_events:
                from engine.visible_inset_cells import inset_edits
                added, removed = inset_edits(image.size, metric_camera, measured_patches, inset_events, cells)
                cells = (set(cells) | added) - removed
                status.update(placed_blocks=len(cells), inset_placement=dict(added=len(added), removed=len(removed),
                    source='photographed backs and occluding cells inside measured rims'))
            if pre_inset is not None and cells and joint_foreground:
                from engine import pavement_opening, window_wall_recheck
                from engine.pavement_obstructions import window_obstructions
                # A flush platform can share the pavement grid. Keep its surface
                # and counts, and test only elevated cubes obstructing those rays.
                matched = set(pavement_opening.detect(image, pre_inset, status))
                if matched:
                    regions = [part for part in pre_inset if part['id'] in matched]
                    blocked = window_obstructions(image.size, status, regions, cells)
                    cells = set(cells) - blocked
                    restored = window_wall_recheck.recover(image, placement_parts, status, matched, cells)
                    cells.update(restored)
                    status.update(placed_blocks=len(cells), pavement_openings=dict(
                        surfaces=sorted(matched), removed=sorted(blocked), recovered_walls=sorted(restored)))
                elif (counter == texture_guided_count.METHOD and material_match in (None,'atlas')
                      and not (inset_events or phase_events or count_confirmed_outlines)
                      and status.get('foreground_depth',{}).get('method') == 'joint'):
                    from engine import smooth_extent_placement
                    source_camera = metric_camera if metric_camera is not None else dict(
                        P=status['P'], scale_status=status['camera_status'], reason=status['reason'])
                    additions, regions = smooth_extent_placement.recover(
                        image, version, placement_parts, set(cells), source_camera, camera_atlas_path)
                    if regions:
                        cells = set(cells) | additions
                        status.update(placed_blocks=len(cells), smooth_extent_placement=dict(
                            candidate_patches=len(regions), added=len(additions)))
                    if common_wall_regions:
                        # Freeze all cells already supported by the ordinary smooth-placement pass.
                        # Reproject this independently confirmed wall patch as an add-only trial.
                        added = smooth_extent_placement.recover_from_regions(
                            image,version,placement_parts,set(cells),source_camera,camera_atlas_path,common_wall_regions)
                        cells = set(cells) | added
                        status.update(placed_blocks=len(cells),common_wall_placement=dict(
                            added=len(added),candidate_patches=len(common_wall_regions)))
                    from engine import counted_ground_corner
                    added, removed, planes = counted_ground_corner.edits(
                        image, placement_parts, placement_counts, status, set(cells))
                    if added or removed:
                        cells = (set(cells) - removed) | added
                        status.update(placed_blocks=len(cells), counted_ground_corner=dict(
                            added=sorted(added), removed=sorted(removed), planes=planes))
            if pre_inset is not None and joint_foreground and status.get('foreground_depth',{}).get('method') == 'joint':
                from engine import exterior_foreground
                background = exterior_foreground.exterior(image)
                # Horizon evidence constrains the scene fit independently of its structural outline.
                roof_parts = exterior_only if exterior_only is not None else patches
                removed = exterior_foreground.misplaced(image, roof_parts, status, cells, background)
                added = set()
                if exterior_events:
                    added, shelf = exterior_foreground.roof_edits(
                        image, scene_context, roof_parts, exterior_events, status, set(cells), background)
                    removed |= shelf
                if removed or added:
                    cells = (set(cells) - removed) | added
                    status.update(placed_blocks=len(cells), exterior_background_removed=sorted(removed),
                                  exterior_roof_recovered=sorted(added))
            material = 'smooth_stone'
            blocks = {tuple(map(int, pos)): material for pos in cells}
            rows = [dict(x=x, y=y, z=z, block_id=material) for x,y,z in sorted(blocks)]
            placement = dict(version=version, camera_status=status['camera_status'],
                             coordinates='local block grid relative to the placement function origin',
                             material=material, blocks=rows)
            write_pack(out/'reconstruction.zip', blocks, JAVA_26_3_FORMAT)
            (out/'placements.json').write_text(json.dumps(placement, indent=2))
            methods['placement'] = 'camera-supported-visible-planes'
            manifest['placement'] = dict(status, material=material)
            manifest['artifacts'].update(placements='placements.json', datapack='reconstruction.zip')
            manifest['artifact'] = 'reconstruction.zip'
    (out/'build-manifest.json').write_text(json.dumps(manifest, indent=2))
