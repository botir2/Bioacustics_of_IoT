package com.example.acousticapp.model

data class DetectionEvent(
    val id: String = "",
    val deviceName: String = "",
    val eventType: String = "",
    val timestamp: String = "",
    val confidence: Int = 0,
    val status: String = ""
)
