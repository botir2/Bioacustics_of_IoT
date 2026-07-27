package com.example.acousticapp.repository

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.URL

class WeatherRepository {

    suspend fun getHobartWeather(): String = withContext(Dispatchers.IO) {
        try {
            val url = "https://api.open-meteo.com/v1/forecast?latitude=-42.8821&longitude=147.3272&current=temperature_2m,weather_code,wind_speed_10m&timezone=Australia%2FHobart"
            val response = URL(url).readText()
            val json = JSONObject(response)
            val current = json.getJSONObject("current")
            
            val temp = current.getDouble("temperature_2m").toInt()
            val wind = current.getDouble("wind_speed_10m").toInt()
            val code = current.getInt("weather_code")
            
            val condition = mapWeatherCode(code)
            
            "$condition • $temp°C • Wind $wind km/h"
        } catch (e: Exception) {
            e.printStackTrace()
            "Weather unavailable"
        }
    }

    private fun mapWeatherCode(code: Int): String {
        return when (code) {
            0 -> "Clear sky"
            1, 2, 3 -> "Partly cloudy"
            45, 48 -> "Fog"
            51, 53, 55 -> "Drizzle"
            61, 63, 65 -> "Rain"
            71, 73, 75 -> "Snow"
            80, 81, 82 -> "Rain showers"
            95 -> "Thunderstorm"
            else -> "Weather update"
        }
    }
}
