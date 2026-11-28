import com.mojang.blaze3d.pipeline.RenderTarget;
import net.minecraft.client.Camera;
import net.minecraft.core.BlockPos;
import net.minecraft.client.Minecraft;

/** Version-specific client accessors. The experiment itself lives in PaletteCapture. */
final class ClientView {
    static void hideHud(Minecraft mc) { mc.options.hideGui=true; }

    static boolean meshReady(Minecraft mc, BlockPos position) {
        return mc.levelRenderer.isSectionCompiledAndVisible(position);
    }

    static boolean hidden(Minecraft mc) { return mc.options.hideGui; }
    static boolean obstructed(Minecraft mc) { return mc.screen!=null || mc.getOverlay()!=null; }
    static Camera camera(Minecraft mc) { return mc.gameRenderer.getMainCamera(); }
    static RenderTarget renderTarget(Minecraft mc) { return mc.getMainRenderTarget(); }
}
