package com.nlu.assistant

import kotlin.system.exitProcess

/**
 * Main entry point for T5 NLU inference application.
 */
fun main(args: Array<String>) {
    println()
    println("═".repeat(80))
    println("T5 NLU Assistant - Kotlin/ONNX Runtime")
    println("Optimized for Snapdragon X Elite NPU")
    println("═".repeat(80))
    println()

    // Parse command line arguments
    val config = parseArguments(args)

    if (config.showHelp) {
        printUsage()
        exitProcess(0)
    }

    // Load model
    println("Loading model from: ${config.modelPath}")
    println("NPU mode: ${if (config.useNPU) "enabled" else "disabled"}")
    println()

    val model = try {
        T5NLUModel.load(config.modelPath, config.useNPU)
    } catch (e: Exception) {
        println("✗ Error loading model: ${e.message}")
        e.printStackTrace()
        exitProcess(1)
    }

    // Run in appropriate mode
    when {
        config.testMode -> runTests(model)
        config.benchmarkMode -> runBenchmark(model, config.numIterations)
        config.inputText != null -> runSingleInference(model, config.inputText)
        else -> runInteractiveMode(model)
    }

    model.close()
}

/**
 * Configuration from command line arguments.
 */
data class Config(
    val modelPath: String = "../models/onnx/t5_nlu_full",
    val useNPU: Boolean = false,
    val testMode: Boolean = false,
    val benchmarkMode: Boolean = false,
    val numIterations: Int = 100,
    val inputText: String? = null,
    val showHelp: Boolean = false
)

/**
 * Parse command line arguments.
 */
fun parseArguments(args: Array<String>): Config {
    var modelPath = "../models/onnx/t5_nlu_full"
    var useNPU = false
    var testMode = false
    var benchmarkMode = false
    var numIterations = 100
    var inputText: String? = null
    var showHelp = false

    var i = 0
    while (i < args.size) {
        when (args[i]) {
            "-m", "--model" -> {
                modelPath = args.getOrNull(++i) ?: modelPath
            }
            "--npu", "--qnn" -> {
                useNPU = true
            }
            "-t", "--test" -> {
                testMode = true
            }
            "-b", "--benchmark" -> {
                benchmarkMode = true
            }
            "-n", "--iterations" -> {
                numIterations = args.getOrNull(++i)?.toIntOrNull() ?: numIterations
            }
            "-i", "--input" -> {
                inputText = args.getOrNull(++i)
            }
            "-h", "--help" -> {
                showHelp = true
            }
            else -> {
                println("Unknown argument: ${args[i]}")
            }
        }
        i++
    }

    return Config(
        modelPath, useNPU, testMode, benchmarkMode,
        numIterations, inputText, showHelp
    )
}

/**
 * Print usage information.
 */
fun printUsage() {
    println("""
        Usage: kotlin-nlu [OPTIONS]

        Options:
          -m, --model <path>      Path to ONNX model directory
                                  (default: ../models/onnx/t5_nlu_full)
          --npu, --qnn            Enable Snapdragon NPU (requires onnxruntime-qnn)
          -t, --test              Run test suite with example utterances
          -b, --benchmark         Run performance benchmark
          -n, --iterations <num>  Number of iterations for benchmark (default: 100)
          -i, --input <text>      Run single inference on input text
          -h, --help              Show this help message

        Examples:
          # Interactive mode (default)
          kotlin-nlu

          # Run with NPU acceleration
          kotlin-nlu --npu

          # Run test suite
          kotlin-nlu --test

          # Single inference
          kotlin-nlu -i "set a timer for 5 minutes"

          # Benchmark performance
          kotlin-nlu --benchmark --iterations 50

        For Windows with Snapdragon X Elite, use --npu flag to enable NPU acceleration.
    """.trimIndent())
}

/**
 * Run test suite.
 */
fun runTests(model: T5NLUModel) {
    println("═".repeat(80))
    println("RUNNING TEST SUITE")
    println("═".repeat(80))
    println()

    val testCases = listOf(
        "set a timer for 5 minutes",
        "what is the weather",
        "open YouTube",
        "volume up",
        "remind me to eat dinner at 6 pm",
        "search Google for cats",
        "turn brightness down",
        "play music",
        "what time is it",
        "tell me a joke"
    )

    val results = mutableListOf<NLUResult>()

    testCases.forEachIndexed { index, text ->
        println("${index + 1}. Input: \"$text\"")
        val result = model.predict(text)

        println("   Intent: ${result.intent}")
        if (result.params.isNotEmpty()) {
            println("   Params: ${result.params}")
        }
        println("   Inference time: ${result.inferenceTimeMs}ms")
        if (result.parseError != null) {
            println("   ⚠ Parse error: ${result.parseError}")
        }
        println()

        results.add(result)
    }

    // Summary
    val avgTime = results.map { it.inferenceTimeMs }.average()
    val successRate = results.count { it.intent != "PARSE_ERROR" } * 100.0 / results.size

    println("═".repeat(80))
    println("TEST SUMMARY")
    println("═".repeat(80))
    println("Total tests: ${results.size}")
    println("Success rate: ${"%.1f".format(successRate)}%")
    println("Average inference time: ${"%.2f".format(avgTime)}ms")
    println("Min time: ${results.minOf { it.inferenceTimeMs }}ms")
    println("Max time: ${results.maxOf { it.inferenceTimeMs }}ms")
    println("═".repeat(80))
}

/**
 * Run performance benchmark.
 */
fun runBenchmark(model: T5NLUModel, iterations: Int) {
    println("═".repeat(80))
    println("PERFORMANCE BENCHMARK")
    println("═".repeat(80))
    println("Iterations: $iterations")
    println()

    val testText = "set a timer for 5 minutes"
    val times = mutableListOf<Long>()

    // Warmup
    print("Warming up...")
    repeat(5) {
        model.predict(testText)
    }
    println(" done")
    println()

    // Benchmark
    println("Running benchmark...")
    repeat(iterations) { i ->
        if ((i + 1) % 10 == 0) {
            print(".")
        }
        val result = model.predict(testText)
        times.add(result.inferenceTimeMs)
    }
    println()
    println()

    // Statistics
    val avgTime = times.average()
    val minTime = times.minOrNull() ?: 0
    val maxTime = times.maxOrNull() ?: 0
    val medianTime = times.sorted()[times.size / 2]
    val stdDev = kotlin.math.sqrt(
        times.map { (it - avgTime).let { diff -> diff * diff } }.average()
    )

    println("═".repeat(80))
    println("BENCHMARK RESULTS")
    println("═".repeat(80))
    println("Iterations: $iterations")
    println("Average: ${"%.2f".format(avgTime)}ms")
    println("Median: ${medianTime}ms")
    println("Min: ${minTime}ms")
    println("Max: ${maxTime}ms")
    println("Std Dev: ${"%.2f".format(stdDev)}ms")
    println()
    println("Throughput: ${"%.2f".format(1000.0 / avgTime)} queries/second")
    println("═".repeat(80))
}

/**
 * Run single inference.
 */
fun runSingleInference(model: T5NLUModel, text: String) {
    println("Input: \"$text\"")
    println()

    val result = model.predict(text)

    println("Intent: ${result.intent}")
    if (result.params.isNotEmpty()) {
        println("Params:")
        result.params.forEach { (key, value) ->
            println("  - $key: $value")
        }
    } else {
        println("Params: (none)")
    }
    println()
    println("Inference time: ${result.inferenceTimeMs}ms")
    println()
    println("Raw output:")
    println(result.rawOutput)

    if (result.parseError != null) {
        println()
        println("⚠ Parse error: ${result.parseError}")
    }
}

/**
 * Run interactive mode.
 */
fun runInteractiveMode(model: T5NLUModel) {
    println("═".repeat(80))
    println("INTERACTIVE MODE")
    println("Type your message or 'quit' to exit")
    println("═".repeat(80))
    println()

    while (true) {
        print(">>> ")
        val input = readLine()?.trim() ?: continue

        if (input.isEmpty()) continue
        if (input.lowercase() in listOf("quit", "exit", "q")) {
            println("Exiting...")
            break
        }

        try {
            val result = model.predict(input)

            println("Intent: ${result.intent}")
            if (result.params.isNotEmpty()) {
                println("Params: ${result.params}")
            } else {
                println("Params: (none)")
            }
            println("Time: ${result.inferenceTimeMs}ms")

            if (result.parseError != null) {
                println("⚠ ${result.parseError}")
            }
        } catch (e: Exception) {
            println("✗ Error: ${e.message}")
        }

        println()
    }
}
