import com.google.gson.Gson;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.TreeMap;
import net.minecraft.SharedConstants;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.block.state.properties.Property;

public class CoreRegistry {
    public static void main(String[] args) throws Exception {
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();
        var blocks = new TreeMap<String,Object>();
        for (var block : BuiltInRegistries.BLOCK) {
            var properties = new TreeMap<String,String>();
            block.defaultBlockState().getValues().map(Property.Value::toString).forEach(pair -> {
                var parts = pair.split("=", 2);
                properties.put(parts[0], parts[1]);
            });
            blocks.put(BuiltInRegistries.BLOCK.getKey(block).getPath(), properties);
        }
        Files.writeString(Path.of(args[0]), new Gson().toJson(blocks));
    }
}
