plugins {
    kotlin("jvm") version "1.9.20"
    application
}

group = "com.nlu"
version = "1.0.0"

repositories {
    mavenCentral()
}

dependencies {
    // ONNX Runtime for Snapdragon NPU
    implementation("com.microsoft.onnxruntime:onnxruntime:1.16.3")

    // HuggingFace Tokenizers for proper T5 tokenization
    implementation("ai.djl.huggingface:tokenizers:0.27.0")

    // JSON parsing
    implementation("org.json:json:20231013")
    implementation("com.google.code.gson:gson:2.10.1")

    // Kotlin coroutines for async operations
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")

    // Logging
    implementation("org.slf4j:slf4j-simple:2.0.9")

    // Testing
    testImplementation(kotlin("test"))
}

application {
    mainClass.set("com.nlu.assistant.MainKt")
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(17)
}

// Create a fat JAR with all dependencies
tasks.jar {
    manifest {
        attributes["Main-Class"] = "com.nlu.assistant.MainKt"
    }

    duplicatesStrategy = DuplicatesStrategy.EXCLUDE

    from(configurations.runtimeClasspath.get().map { if (it.isDirectory) it else zipTree(it) })
}
