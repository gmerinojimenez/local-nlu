package com.nlu.assistant

import ai.djl.huggingface.tokenizers.Encoding
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import java.nio.file.Files
import java.nio.file.Paths

/**
 * T5 tokenizer using HuggingFace tokenizers library.
 * Properly handles SentencePiece tokenization.
 */
class T5Tokenizer private constructor(
    private val tokenizer: HuggingFaceTokenizer
) {

    /**
     * Encode text to token IDs.
     */
    fun encode(text: String, maxLength: Int = 128): IntArray {
        val encoding = tokenizer.encode(text)
        val tokens = encoding.ids // This is LongArray

        // Convert to IntArray and pad or truncate to maxLength
        return when {
            tokens.size > maxLength -> tokens.copyOf(maxLength).map { it.toInt() }.toIntArray()
            tokens.size < maxLength -> {
                val padded = IntArray(maxLength) { 0 } // PAD token is 0
                tokens.forEachIndexed { index, value -> padded[index] = value.toInt() }
                padded
            }
            else -> tokens.map { it.toInt() }.toIntArray()
        }
    }

    /**
     * Decode token IDs to text.
     */
    fun decode(tokenIds: IntArray, skipSpecialTokens: Boolean = true): String {
        // Convert to LongArray as required by tokenizer
        val longIds = tokenIds.map { it.toLong() }.toLongArray()
        return tokenizer.decode(longIds, skipSpecialTokens)
    }

    fun close() {
        tokenizer.close()
    }

    companion object {
        /**
         * Load tokenizer from model directory.
         * Uses HuggingFace tokenizers library for proper SentencePiece tokenization.
         */
        fun loadFromPath(modelPath: String): T5Tokenizer {
            val basePath = Paths.get(modelPath)

            // Try to find tokenizer in multiple locations
            val tokenizerPaths = listOf(
                basePath.parent.resolve("tokenizer"),
                basePath,
                basePath.parent.parent.resolve("best_model")
            )

            for (path in tokenizerPaths) {
                // Look for tokenizer.json (HuggingFace format)
                val tokenizerJsonPath = path.resolve("tokenizer.json")
                if (Files.exists(tokenizerJsonPath)) {
                    println("Loading tokenizer from: $path")
                    val tokenizer = HuggingFaceTokenizer.newInstance(tokenizerJsonPath)
                    println("✓ Using HuggingFace tokenizer with SentencePiece")
                    return T5Tokenizer(tokenizer)
                }

                // Fallback: check for tokenizer_config.json (will need conversion)
                val configPath = path.resolve("tokenizer_config.json")
                if (Files.exists(configPath)) {
                    println("Found tokenizer config at: $path")
                    println("⚠ tokenizer.json not found. This is required for HuggingFace tokenizers.")
                    println("  Please convert your tokenizer to tokenizer.json format:")
                    println("  python -c \"from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('$path'); t.save_pretrained('$path', legacy_format=False)\"")
                    throw IllegalArgumentException("tokenizer.json is required but not found in $path")
                }
            }

            throw IllegalArgumentException(
                "Tokenizer not found. Tried paths: ${tokenizerPaths.joinToString()}\n" +
                "Please ensure tokenizer.json exists in one of these locations."
            )
        }
    }
}
