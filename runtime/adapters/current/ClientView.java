import com.mojang.blaze3d.pipeline.RenderTarget;
import net.minecraft.client.Camera;
import net.minecraft.client.Minecraft;

/** Version-specific client accessors. The experiment itself lives in PaletteCapture. */
final class ClientView {
    static void hideHud(Minecraft mc) {
        if (!mc.gui.hud.isHidden()) mc.gui.hud.toggle();
    }

    static boolean hidden(Minecraft mc) { return mc.gui.hud.isHidden(); }
    static boolean obstructed(Minecraft mc) { return mc.gui.screen()!=null || mc.gui.overlay()!=null; }
    static Camera camera(Minecraft mc) { return mc.gameRenderer.mainCamera(); }
    static RenderTarget renderTarget(Minecraft mc) { return mc.gameRenderer.mainRenderTarget(); }
}
