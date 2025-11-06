package com.nlu.assistant

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.gson.Gson
import com.google.gson.JsonSyntaxException
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.io.path.exists

/**
 * T5-based NLU model for intent classification and parameter extraction.
 * Uses ONNX Runtime for efficient inference on Snapdragon NPU.
 */
class T5NLUModel(
    private val modelPath: String,
    private val useQNN: Boolean = false
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var encoderSession: OrtSession
    private lateinit var decoderSession: OrtSession
    private val tokenizer: T5Tokenizer
    private val gson = Gson()

    init {
        println("=" .repeat(80))
        println("T5 NLU Model - ONNX Runtime")
        println("=" .repeat(80))

        // Load tokenizer
        tokenizer = T5Tokenizer.loadFromPath(modelPath)
        println("✓ Tokenizer loaded")

        // Create session options
        val sessionOptions = OrtSession.SessionOptions()

        if (useQNN) {
            println("🚀 Attempting to use QNN Execution Provider (Snapdragon NPU)...")
            // QNN provider configuration for ONNX Runtime
            // Note: QNN support in Java API is limited
            // For full NPU support, use native libraries or Python
            try {
                // QNN provider would be added here when available
                // Currently falling back to CPU as Java API doesn't expose QNN directly
                println("⚠ QNN provider not yet supported in Java API")
                println("  Using CPU. For NPU, consider using Python implementation")
            } catch (e: Exception) {
                println("⚠ QNN not available: ${e.message}")
                println("  Falling back to CPU")
            }
        } else {
            println("💻 Using CPU Execution Provider")
        }

        // Load ONNX models
        val basePath = Paths.get(modelPath)

        val encoderPath = basePath.resolve("encoder_model.onnx")
        val decoderPath = basePath.resolve("decoder_model.onnx")

        if (!encoderPath.exists() || !decoderPath.exists()) {
            throw IllegalArgumentException(
                "ONNX models not found at $modelPath. " +
                "Please run: python scripts/export_to_onnx.py models/best_model"
            )
        }

        println("\nLoading ONNX models...")
        encoderSession = env.createSession(encoderPath.toString(), sessionOptions)
        decoderSession = env.createSession(decoderPath.toString(), sessionOptions)
        println("✓ Encoder and Decoder loaded")

        println("\nModel ready for inference!")
        println("=" .repeat(80))
    }

    /**
     * Predict intent and parameters from input text.
     */
    fun predict(text: String, maxLength: Int = 256, numBeams: Int = 4): NLUResult {
        val startTime = System.currentTimeMillis()

        // Format input
        val inputText = "nlu: $text"

        // Tokenize
        val inputIds = tokenizer.encode(inputText, maxLength = 128)

        // Run encoder
        val encoderOutputs = runEncoder(inputIds)

        // Run decoder with greedy search (simplified from beam search)
        val outputTokens = runDecoder(
            encoderOutputs,
            inputIds.size,
            maxLength
        )

        // Decode output
        val rawOutput = tokenizer.decode(outputTokens)

        val inferenceTime = System.currentTimeMillis() - startTime

        // Parse JSON
        return parseOutput(rawOutput, inferenceTime)
    }

    private fun runEncoder(inputIds: IntArray): FloatArray {
        val batchSize = 1
        val seqLength = inputIds.size

        // Create input tensors
        val inputIdsTensor = OnnxTensor.createTensor(
            env,
            arrayOf(inputIds.map { it.toLong() }.toLongArray())
        )

        val attentionMask = IntArray(seqLength) { 1 }
        val attentionMaskTensor = OnnxTensor.createTensor(
            env,
            arrayOf(attentionMask.map { it.toLong() }.toLongArray())
        )

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to attentionMaskTensor
        )

        // Run encoder
        val outputs = encoderSession.run(inputs)
        val encoderOutput = outputs[0].value as Array<*>

        inputIdsTensor.close()
        attentionMaskTensor.close()
        outputs.close()

        // Flatten the output
        @Suppress("UNCHECKED_CAST")
        val output3D = encoderOutput as Array<Array<FloatArray>>
        return output3D[0].flatMap { it.toList() }.toFloatArray()
    }

    private fun runDecoder(
        encoderHiddenStates: FloatArray,
        encoderSeqLength: Int,
        maxLength: Int
    ): IntArray {
        val outputTokens = mutableListOf(0) // Start with PAD token

        for (step in 0 until maxLength) {
            // Create decoder input
            val decoderInputIds = outputTokens.toIntArray()
            val decoderInputIdsTensor = OnnxTensor.createTensor(
                env,
                arrayOf(decoderInputIds.map { it.toLong() }.toLongArray())
            )

            // Encoder attention mask
            val encoderAttentionMask = IntArray(encoderSeqLength) { 1 }
            val encoderAttentionMaskTensor = OnnxTensor.createTensor(
                env,
                arrayOf(encoderAttentionMask.map { it.toLong() }.toLongArray())
            )

            // Reshape encoder hidden states (T5-base uses 768 dimensions)
            val hiddenDim = 768
            val hiddenStatesTensor = OnnxTensor.createTensor(
                env,
                Array(1) { Array(encoderSeqLength) { i ->
                    FloatArray(hiddenDim) { j ->
                        encoderHiddenStates[i * hiddenDim + j]
                    }
                }}
            )

            val inputs = mapOf(
                "input_ids" to decoderInputIdsTensor,
                "encoder_attention_mask" to encoderAttentionMaskTensor,
                "encoder_hidden_states" to hiddenStatesTensor
            )

            // Run decoder
            val outputs = decoderSession.run(inputs)
            val logits = outputs[0].value as Array<*>

            decoderInputIdsTensor.close()
            encoderAttentionMaskTensor.close()
            hiddenStatesTensor.close()
            outputs.close()

            // Get next token (greedy)
            @Suppress("UNCHECKED_CAST")
            val logits3D = logits as Array<Array<FloatArray>>
            val lastLogits = logits3D[0].last()
            val nextToken = lastLogits.indices.maxByOrNull { lastLogits[it] } ?: 0

            outputTokens.add(nextToken)

            // Stop if EOS token (1 for T5)
            if (nextToken == 1) break
        }

        return outputTokens.toIntArray()
    }

    private fun parseOutput(rawOutput: String, inferenceTime: Long): NLUResult {
        // Try to parse as JSON
        try {
            val result = gson.fromJson(rawOutput, NLUJsonResult::class.java)
            return NLUResult(
                intent = result.intent,
                params = result.params,
                rawOutput = rawOutput,
                inferenceTimeMs = inferenceTime,
                parseError = null
            )
        } catch (e: JsonSyntaxException) {
            // Fallback: extract intent with regex
            val intentRegex = """"intent":\s*"([^"]+)"""".toRegex()
            val intentMatch = intentRegex.find(rawOutput)
            val intent = intentMatch?.groupValues?.get(1) ?: "PARSE_ERROR"

            return NLUResult(
                intent = intent,
                params = emptyMap(),
                rawOutput = rawOutput,
                inferenceTimeMs = inferenceTime,
                parseError = "Failed to parse JSON: ${e.message}"
            )
        }
    }

    override fun close() {
        encoderSession.close()
        decoderSession.close()
    }

    companion object {
        /**
         * Load model from ONNX export directory.
         */
        fun load(modelPath: String, useQNN: Boolean = false): T5NLUModel {
            return T5NLUModel(modelPath, useQNN)
        }
    }
}

/**
 * Result of NLU inference.
 */
data class NLUResult(
    val intent: String,
    val params: Map<String, Any>,
    val rawOutput: String,
    val inferenceTimeMs: Long,
    val parseError: String?
) {
    fun toJson(): String {
        return """
            {
              "intent": "$intent",
              "params": ${Gson().toJson(params)},
              "inference_time_ms": $inferenceTimeMs,
              "raw_output": "$rawOutput"
            }
        """.trimIndent()
    }
}

/**
 * Internal JSON result structure.
 */
private data class NLUJsonResult(
    val intent: String,
    val params: Map<String, Any>
)
