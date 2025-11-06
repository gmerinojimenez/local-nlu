# Tokenizer Limitation

## Current Issue

The Kotlin implementation uses a **simplified whitespace tokenizer** which only knows a few words:

```kotlin
val vocab = mutableMapOf(
    "<pad>" to 0,
    "</s>" to 1,
    "<unk>" to 2,
    "nlu" to 3,
    ":" to 4,
    "set" to 10,
    "timer" to 11,
    // ... only ~20 words
)
```

This causes the model to output repeated "nlu" tokens because unknown words get mapped to token ID 3 ("nlu").

## Solutions

### Option 1: Use Python for Real Inference (Recommended)

The Python implementation has full SentencePiece tokenizer support:

```bash
# This works perfectly
python scripts/test_checkpoint_optimum.py models/onnx/t5_nlu_full
```

Output:
```
1. Input: "set a timer for 5 minutes"
   Raw model output: "intent": "SET_TIMER", "params": "timer_duration": "amount": 5, "unit": "min"
   Inference time: 651.65ms
   Parsed Intent: SET_TIMER
```

### Option 2: Integrate SentencePiece in Kotlin

For production Kotlin deployment, integrate SentencePiece:

```gradle
// In build.gradle.kts
dependencies {
    implementation("com.github.google.sentencepiece:sentencepiece:0.1.99")
}
```

Then load the actual model:

```kotlin
// Load SentencePiece model from spiece.model file
val processor = SentencePieceProcessor()
processor.load(modelPath.resolve("spiece.model").toString())

// Encode
val tokens = processor.encode(text)

// Decode
val text = processor.decode(tokens)
```

### Option 3: Pre-compute Vocabulary

Load the full vocabulary from `added_tokens.json`:

```kotlin
// Load ~32k vocabulary entries
val vocabFile = tokenizerPath.resolve("added_tokens.json")
val vocabJson = Files.readString(vocabFile)
val vocab: Map<String, Int> = gson.fromJson(vocabJson, ...)
```

## Current State

- ✅ **Kotlin code structure**: Complete and working
- ✅ **ONNX Runtime integration**: Functional
- ✅ **CLI and modes**: All working
- ⚠️ **Tokenizer**: Simplified (insufficient for real inference)

## Recommendation

### For Development/Testing
Use the **Python implementation** which has full tokenizer support:
```bash
python scripts/test_checkpoint_optimum.py models/onnx/t5_nlu_full
```

### For Production JVM Deployment
1. Use the Kotlin code structure (already complete)
2. Integrate proper tokenizer:
   - Option A: SentencePiece Java bindings
   - Option B: Call Python tokenizer via subprocess
   - Option C: Pre-tokenize queries and use token IDs directly

### For Snapdragon NPU
Use Python implementation:
```bash
python scripts/test_checkpoint_optimum.py models/onnx/t5_nlu_full --npu
```

## What Works

The Kotlin implementation **works perfectly** for everything except tokenization:

✅ Gradle build system
✅ ONNX Runtime integration
✅ Encoder/decoder inference
✅ CLI with all modes
✅ Performance measurement
✅ JSON parsing
✅ 84MB fat JAR
⚠️ Tokenization (needs proper library)

## Next Steps

Choose based on your use case:

**Quick Testing/NPU**: Use Python
```bash
python scripts/test_checkpoint_optimum.py models/onnx/t5_nlu_full
```

**Production JVM**: Add SentencePiece to Kotlin
```gradle
implementation("com.github.google.sentencepiece:sentencepiece:0.1.99")
```

**Hybrid Approach**:
- Use Python for inference
- Kotlin for application logic
- Call Python via REST API or subprocess

---

**Summary**: The Kotlin ONNX Runtime implementation is **structurally complete** but needs a proper tokenizer library for actual inference. For real use, stick with the Python implementation or integrate SentencePiece into the Kotlin code.
