package com.example.acousticapp.util

import java.util.Locale

object LocationUtils {

    fun formatLatLng(latitude: Double, longitude: Double): String {
        return String.format(Locale.getDefault(), "%.6f, %.6f", latitude, longitude)
    }

    fun isValidLocation(latitude: Double, longitude: Double): Boolean {
        return latitude in -90.0..90.0 && longitude in -180.0..180.0
    }

    fun buildGpsAddressPlaceholder(latitude: Double, longitude: Double): String {
        return "Address for (${formatLatLng(latitude, longitude)})"
    }
}
