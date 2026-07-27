package com.example.acousticapp.repository

import com.example.acousticapp.model.*

class DetectionRepository {
    // TODO: Replace mock data with Firestore data
    fun getMockDetections(): List<DetectionEvent> = listOf(
        DetectionEvent("1", "Acoustic Device 001", "Glass Break", "Today, 10:30 AM", 92, "Alert"),
        DetectionEvent("2", "Acoustic Device 003", "Loud Sound", "Today, 09:15 AM", 78, "Warning"),
        DetectionEvent("3", "Acoustic Device 001", "Siren", "Yesterday, 08:45 PM", 95, "Alert"),
        DetectionEvent("4", "Acoustic Device 002", "Normal Activity", "Yesterday, 06:10 PM", 65, "Normal")
    )
}


