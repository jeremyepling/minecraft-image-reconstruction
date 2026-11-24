import com.google.gson.GsonBuilder;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.client.Camera;
import net.minecraft.client.Minecraft;
import net.minecraft.client.Screenshot;
import net.minecraft.client.main.Main;
import net.minecraft.server.MinecraftServer;
import org.joml.Matrix4f;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.ArrayList;
import java.util.Map;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/** Start the unmodified client, drive only this process, and persist its native framebuffer. */
public final class PaletteCapture {
    private record View(String id, double x, double feetY, double z, double[] target, int page) {
    }

    public static void main(String[] args) throws Exception {
        Thread driver = new Thread(() -> {
            try {
                capture();
            } catch (Throwable e) {
                e.printStackTrace();
                System.exit(2);
            }
        }, "palette-lab-capture");
        driver.setDaemon(true);
        driver.start();
        Main.main(args);
    }

    private static void command(MinecraftServer server, String text) {
        server.getCommands().performPrefixedCommand(server.createCommandSourceStack(), text);
    }

    private static void capture() throws Exception {
        Path out = Path.of(System.getProperty("palette.capture.dir"));
        Files.createDirectories(out);
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(4);
        while (Minecraft.getInstance() == null || Minecraft.getInstance().player == null
                  || Minecraft.getInstance().getSingleplayerServer() == null) {
            if (System.nanoTime() > deadline) throw new IllegalStateException("Private world failed to open");
            Thread.sleep(200);
        }
        Minecraft mc = Minecraft.getInstance();
        MinecraftServer server = mc.getSingleplayerServer();
        mc.submit(() -> {
            mc.options.pauseOnLostFocus = false;
            mc.options.bobView().set(false);
            mc.options.fov().set(55);
            mc.options.fovEffectScale().set(0.0);
            ClientView.hideHud(mc);
        }).get(20, TimeUnit.SECONDS);
        server.submit(() -> {
            command(server, "forceload add -32 -16 31 15");
            command(server, "gamemode spectator @a");
            command(server, "time set noon");
            command(server, "weather clear");
            command(server, "gamerule minecraft:advance_time false");
            command(server, "gamerule minecraft:advance_weather false");
        }).get(60, TimeUnit.SECONDS);
        Thread.sleep(2500);
        Path viewsPath = out.getParent().resolve("views.json");
        if (!Files.isRegularFile(viewsPath)) throw new IllegalStateException("Missing views.json");
        Gson gson = new Gson();
        View[] selected = gson.fromJson(Files.readString(viewsPath), View[].class);
        JsonObject truth = gson.fromJson(Files.readString(out.getParent().resolve("layout.json")), JsonObject.class);
        var ids = BuiltInRegistries.BLOCK.keySet().stream().map(Object::toString).sorted().toList();
        Files.writeString(out.getParent().resolve("runtime-block-registry.json"),gson.toJson(ids));
        var built = new java.util.HashSet<Integer>();
        for (View v : selected) {
            double eyeY = v.feetY() + 1.62;
            double dx = v.target()[0] - v.x(), dz = v.target()[2] - v.z();
            double yaw = Math.toDegrees(Math.atan2(-dx,dz));
            double pitch = -Math.toDegrees(Math.atan2(v.target()[1]-eyeY,Math.hypot(dx,dz)));
            String tp = String.format(Locale.ROOT, "tp @a %.5f %.5f %.5f %.7f %.7f",v.x(),v.feetY(),v.z(),yaw,pitch);
            server.submit(() -> command(server,tp)).get(30, TimeUnit.SECONDS);
            if (built.add(v.page())) {
                Thread.sleep(1000);
                server.submit(() -> {
                    for (var entry : truth.getAsJsonArray("blocks")) {
                        var b=entry.getAsJsonObject();
                        if (b.get("page").getAsInt()==v.page()) {
                            server.overworld().getChunk(b.get("x").getAsInt()>>4,b.get("z").getAsInt()>>4);
                        }
                    }
                    command(server,"function palette_lab:page_"+v.page());
                }).get(60,TimeUnit.SECONDS);
            }
            // Wait for chunk meshes, the initial loading fade and camera interpolation.
            Thread.sleep(v==selected[0] ? 6500 : 2300);
            {
                var rows = server.submit(() -> {
                    var actual = new ArrayList<Map<String,Object>>();
                    for (var entry : truth.getAsJsonArray("blocks")) {
                        var b=entry.getAsJsonObject();
                        if (b.get("page").getAsInt()!=v.page()) continue;
                        int x=b.get("x").getAsInt(), y=b.get("y").getAsInt(), z=b.get("z").getAsInt();
                        var state=server.overworld().getBlockState(new BlockPos(x,y,z));
                        actual.add(Map.of("x",x,"y",y,"z",z,"block_id",BuiltInRegistries.BLOCK.getKey(state.getBlock()).getPath(),"state",state.toString()));
                    }
                    return actual;
                }).get(30,TimeUnit.SECONDS);
                Files.writeString(out.resolve(v.id()+"-blocks.json"),gson.toJson(rows));
            }
            CompletableFuture<Void> saved = new CompletableFuture<>();
            mc.submit(() -> {
                Camera c = ClientView.camera(mc);
                if (Math.abs(c.position().x()-v.x())>.03 || Math.abs(c.position().z()-v.z())>.03)
                    throw new IllegalStateException("Teleport not applied: "+c.position());
                if (ClientView.obstructed(mc))
                    throw new IllegalStateException("View obstructed by a screen or overlay");
                var info = new LinkedHashMap<String,Object>();
                info.put("id",v.id());
                info.put("eye_world",new double[]{c.position().x(),c.position().y(),c.position().z()});
                info.put("yaw_pitch_deg",new float[]{c.yRot(),c.xRot()});
                info.put("vertical_fov_deg",c.getFov());
                info.put("view_rotation_projection",c.getViewRotationProjectionMatrix(new Matrix4f()).get(new float[16]));
                info.put("clock_ticks",mc.level.getOverworldClockTime());
                info.put("raining",mc.level.isRaining());
                info.put("thundering",mc.level.isThundering());
                info.put("hud_hidden",ClientView.hidden(mc));
                info.put("minecraft_version",net.minecraft.SharedConstants.getCurrentVersion().id());
                if (mc.level.getOverworldClockTime()%24000!=6000 || mc.level.isRaining() || mc.level.isThundering())
                    throw new IllegalStateException("Weather or clock assertion failed: "+info);
                Screenshot.takeScreenshot(ClientView.renderTarget(mc), frame -> {
                    try (frame) {
                        frame.writeToFile(out.resolve(v.id()+".png"));
                        info.put("image_size",new int[]{frame.getWidth(),frame.getHeight()});
                        Files.writeString(out.resolve(v.id()+".json"),new GsonBuilder().setPrettyPrinting().create().toJson(info));
                        System.out.println("[PALETTE] saved "+v.id()+" "+info.get("clock_ticks"));
                        saved.complete(null);
                    } catch (Throwable err) {
                        saved.completeExceptionally(err);
                    }
                });
            }).get(30, TimeUnit.SECONDS);
            saved.get(30,TimeUnit.SECONDS);
        }
        server.submit(() -> {
            command(server,"gamemode creative @a");
            command(server,"tp @a 0.5 65 29.5 180 0");
            command(server,"forceload remove -32 -16 31 15");
            server.saveEverything(true,true,true);
        }).get(45,TimeUnit.SECONDS);
        mc.execute(mc::stop);
    }
}
