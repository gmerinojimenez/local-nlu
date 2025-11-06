# Kotlin Integration Guide

Complete guide for integrating T5 NLU with Kotlin/JVM applications on Windows with Snapdragon X Elite NPU support.

## Overview

The Kotlin implementation provides:
- ✅ ONNX Runtime integration for NPU acceleration
- ✅ Cross-platform JVM compatibility (Windows, macOS, Linux)
- ✅ Simple API for intent classification and parameter extraction
- ✅ Command-line interface for testing and deployment
- ✅ Optimized for Snapdragon X Elite NPU via QNN

## Quick Start

### 1. Build the Project

```bash
cd kotlin-nlu
./gradlew build

# Or on Windows
gradlew.bat build
```

### 2. Run Interactive Mode

```bash
./run.sh

# Or on Windows
run.bat
```

### 3. Run with NPU (Snapdragon X Elite)

```bash
./run.sh --npu

# Or on Windows
run.bat --npu
```

## Project Structure

```
kotlin-nlu/
├── build.gradle.kts              # Gradle configuration
├── settings.gradle.kts           # Project settings
├── src/main/kotlin/com/nlu/assistant/
│   ├── Main.kt                   # Entry point & CLI
│   ├── T5NLUModel.kt             # ONNX inference engine
│   └── T5Tokenizer.kt            # Tokenization
├── run.sh / run.bat              # Quick run scripts
└── README.md                     # Full documentation
```

## API Usage

### Basic Usage

```kotlin
import com.nlu.assistant.T5NLUModel

fun main() {
    // Load model
    val model = T5NLUModel.load(
        modelPath = "models/onnx/t5_nlu_full",
        useQNN = true  // Enable Snapdragon NPU
    )

    // Run inference
    val result = model.predict("set a timer for 5 minutes")

    println("Intent: ${result.intent}")
    println("Params: ${result.params}")
    println("Time: ${result.inferenceTimeMs}ms")

    // Clean up
    model.close()
}
```

### Advanced Usage

```kotlin
// Batch processing
fun processUserQueries(queries: List<String>) {
    T5NLUModel.load("models/onnx/t5_nlu_full", useQNN = true).use { model ->
        queries.forEach { query ->
            val result = model.predict(query)
            processIntent(result.intent, result.params)
        }
    }
}

// Custom parameters
fun customInference() {
    val model = T5NLUModel.load("models/onnx/t5_nlu_full")

    val result = model.predict(
        text = "what's the weather like",
        maxLength = 256,  // Max generation length
        numBeams = 4      // Beam search width (not yet implemented)
    )

    model.close()
}
```

### Integration with Spring Boot

```kotlin
import org.springframework.stereotype.Service
import javax.annotation.PostConstruct
import javax.annotation.PreDestroy

@Service
class NLUService {
    private lateinit var model: T5NLUModel

    @PostConstruct
    fun init() {
        model = T5NLUModel.load(
            modelPath = System.getenv("NLU_MODEL_PATH") ?: "models/onnx/t5_nlu_full",
            useQNN = System.getenv("USE_NPU")?.toBoolean() ?: false
        )
        println("NLU Model initialized")
    }

    @PreDestroy
    fun cleanup() {
        model.close()
        println("NLU Model closed")
    }

    fun classify(utterance: String): NLUResult {
        return model.predict(utterance)
    }
}

// REST Controller
@RestController
@RequestMapping("/api/nlu")
class NLUController(private val nluService: NLUService) {

    @PostMapping("/classify")
    fun classify(@RequestBody request: ClassifyRequest): ClassifyResponse {
        val result = nluService.classify(request.text)

        return ClassifyResponse(
            intent = result.intent,
            params = result.params,
            confidence = 1.0,  // Add confidence scoring if needed
            inferenceTimeMs = result.inferenceTimeMs
        )
    }
}

data class ClassifyRequest(val text: String)
data class ClassifyResponse(
    val intent: String,
    val params: Map<String, Any>,
    val confidence: Double,
    val inferenceTimeMs: Long
)
```

### Integration with Ktor

```kotlin
import io.ktor.server.application.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.serialization.gson.*
import io.ktor.server.plugins.contentnegotiation.*

fun Application.module() {
    install(ContentNegotiation) {
        gson()
    }

    // Initialize NLU model
    val model = T5NLUModel.load("models/onnx/t5_nlu_full", useQNN = true)

    routing {
        post("/nlu/classify") {
            val request = call.receive<Map<String, String>>()
            val text = request["text"] ?: error("Missing 'text' field")

            val result = model.predict(text)

            call.respond(mapOf(
                "intent" to result.intent,
                "params" to result.params,
                "inference_time_ms" to result.inferenceTimeMs
            ))
        }

        get("/nlu/health") {
            call.respond(mapOf("status" to "healthy"))
        }
    }

    // Cleanup on shutdown
    environment.monitor.subscribe(ApplicationStopped) {
        model.close()
    }
}
```

## Building for Production

### Create Executable JAR

```bash
./gradlew jar

# Output: build/libs/kotlin-nlu-1.0.0.jar
```

### Run the JAR

```bash
# Basic
java -jar kotlin-nlu-1.0.0.jar

# With NPU
java -jar kotlin-nlu-1.0.0.jar --npu

# With custom model path
java -jar kotlin-nlu-1.0.0.jar -m /path/to/models --npu
```

### Windows Service Integration

Create a Windows Service wrapper using [WinSW](https://github.com/winsw/winsw):

```xml
<!-- nlu-service.xml -->
<service>
  <id>T5NLUService</id>
  <name>T5 NLU Service</name>
  <description>Natural Language Understanding Service with Snapdragon NPU</description>
  <executable>java</executable>
  <arguments>-jar kotlin-nlu-1.0.0.jar --npu</arguments>
  <workingdirectory>C:\NLU</workingdirectory>
  <logpath>C:\NLU\logs</logpath>
  <logmode>rotate</logmode>
  <env name="NLU_MODEL_PATH" value="C:\NLU\models\onnx\t5_nlu_full"/>
</service>
```

Install:
```powershell
winsw install nlu-service.xml
winsw start T5NLUService
```

## Performance Optimization

### 1. JVM Tuning

```bash
# Optimize heap size
java -Xmx2G -Xms1G -jar kotlin-nlu-1.0.0.jar

# Enable G1GC for better latency
java -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -jar kotlin-nlu-1.0.0.jar

# GraalVM Native Image (advanced)
native-image -jar kotlin-nlu-1.0.0.jar kotlin-nlu-native
```

### 2. Model Optimization

```kotlin
// Pre-warm the model
fun warmup(model: T5NLUModel) {
    println("Warming up model...")
    repeat(10) {
        model.predict("warmup query")
    }
    println("Warmup complete")
}
```

### 3. Thread Pool for Concurrent Requests

```kotlin
import kotlinx.coroutines.*
import java.util.concurrent.Executors

class NLUService {
    private val model = T5NLUModel.load("models/onnx/t5_nlu_full", useQNN = true)
    private val dispatcher = Executors.newFixedThreadPool(4).asCoroutineDispatcher()

    suspend fun classifyBatch(texts: List<String>): List<NLUResult> = coroutineScope {
        texts.map { text ->
            async(dispatcher) {
                model.predict(text)
            }
        }.awaitAll()
    }

    fun close() {
        model.close()
        dispatcher.close()
    }
}
```

## Deployment Checklist

### For Snapdragon X Elite Windows PC

- [ ] Install Java 17+: `winget install Microsoft.OpenJDK.17`
- [ ] Build JAR: `./gradlew jar`
- [ ] Copy `build/libs/kotlin-nlu-1.0.0.jar` to Windows PC
- [ ] Copy `models/onnx/` directory to Windows PC
- [ ] Test CPU mode: `java -jar kotlin-nlu-1.0.0.jar --test`
- [ ] Test NPU mode: `java -jar kotlin-nlu-1.0.0.jar --npu --test`
- [ ] Run benchmark: `java -jar kotlin-nlu-1.0.0.jar --npu --benchmark`
- [ ] Deploy to production (service/web server)

## Troubleshooting

### Issue: "UnsatisfiedLinkError" on Windows

**Cause:** Missing ONNX Runtime native libraries

**Solution:**
1. Make sure ONNX Runtime JAR includes native binaries
2. Check architecture matches (x64 vs ARM64)
3. Install Visual C++ Redistributable if needed

### Issue: NPU not being used

**Check:**
```kotlin
// Add logging to verify provider
val sessionOptions = OrtSession.SessionOptions()
sessionOptions.addQNN()

// Check available providers
println(OrtEnvironment.getAvailableProviders())
// Should include "QNNExecutionProvider"
```

**Solution:**
- Ensure running on Snapdragon X Elite hardware
- Update to latest Windows 11
- Install Qualcomm NPU drivers

### Issue: Slow first inference

**Normal behavior:** JVM JIT compilation and model loading

**Solution:** Implement warmup routine:
```kotlin
fun main() {
    val model = T5NLUModel.load("models/onnx/t5_nlu_full", useQNN = true)

    // Warmup
    repeat(5) { model.predict("warmup") }

    // Now ready for fast inference
    interactiveMode(model)
}
```

## Performance Benchmarks

Expected performance on different platforms:

| Platform | Backend | Avg Latency | Throughput |
|----------|---------|-------------|------------|
| Intel i7 | CPU | ~450ms | 2.2 req/s |
| Mac M1 | CPU | ~400ms | 2.5 req/s |
| **Snapdragon X Elite** | **NPU** | **~200ms** | **5.0 req/s** |
| Snapdragon X Elite | CPU | ~380ms | 2.6 req/s |

## Comparison: Python vs Kotlin

| Aspect | Python (scripts/) | Kotlin (kotlin-nlu/) |
|--------|------------------|----------------------|
| **Startup Time** | ~2-3s | ~1-2s |
| **Memory Usage** | ~1.5GB | ~800MB |
| **Inference Speed** | Baseline | Similar |
| **Deployment** | Requires Python env | Single JAR file |
| **Integration** | REST API needed | Direct JVM integration |
| **Best For** | Development/Testing | Production deployment |

## Next Steps

1. **Production Integration**: Integrate with your JVM application (Spring Boot, Ktor, etc.)
2. **SentencePiece**: Add proper tokenization using SentencePiece Java bindings
3. **Beam Search**: Implement beam search for better quality
4. **Model Versioning**: Add model version management
5. **Monitoring**: Add metrics and logging for production

## Resources

- [Kotlin NLU README](../kotlin-nlu/README.md) - Full project documentation
- [ONNX Runtime Java API](https://onnxruntime.ai/docs/get-started/with-java.html)
- [Kotlin Coroutines](https://kotlinlang.org/docs/coroutines-overview.html)
- [Snapdragon NPU Docs](https://learn.microsoft.com/en-us/windows/ai/npu-devices/)

## Example Applications

Check the `examples/` directory for:
- REST API server (Spring Boot)
- WebSocket server (Ktor)
- Desktop GUI application (JavaFX)
- Command-line chatbot

---

**Ready to deploy?** Copy the JAR and models to your Windows PC and run with `--npu` flag for Snapdragon acceleration!
