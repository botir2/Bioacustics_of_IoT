package com.example.acousticapp.repository

import com.example.acousticapp.model.AcousticDevice
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.tasks.await

class MapRepository(
    private val db: FirebaseFirestore = FirebaseFirestore.getInstance()
) {
    suspend fun getDeviceLocation(deviceId: String = "device_001"): Result<AcousticDevice> {
        return try {
            val snapshot = db.collection("devices").document(deviceId).get().await()
            val device = snapshot.toObject(AcousticDevice::class.java)
            if (device != null) {
                Result.success(device)
            } else {
                Result.failure(Exception("Device not found"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
