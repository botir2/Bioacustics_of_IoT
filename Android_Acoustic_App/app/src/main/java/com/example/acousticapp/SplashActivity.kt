package com.example.acousticapp

import android.content.Context
import android.content.Intent
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import com.google.firebase.auth.FirebaseAuth

class SplashActivity : AppCompatActivity() {

    private val handler = Handler(Looper.getMainLooper())
    private val fullText = "acoustic"
    private var index = 0
    private var typingCompleted = false
    private var hasNavigated = false

    override fun onCreate(savedInstanceState: Bundle?) {
        // Apply saved theme preference before anything else to avoid recreation loops
        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val isNightMode = prefs.getBoolean("night_mode_enabled", false)
        val targetMode = if (isNightMode) AppCompatDelegate.MODE_NIGHT_YES else AppCompatDelegate.MODE_NIGHT_NO
        
        if (AppCompatDelegate.getDefaultNightMode() != targetMode) {
            AppCompatDelegate.setDefaultNightMode(targetMode)
        }

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        val tv = findViewById<TextView>(R.id.txtTitle)

        val typingRunnable = object : Runnable {
            override fun run() {
                if (index <= fullText.length) {
                    tv.text = fullText.substring(0, index)
                    index++
                    handler.postDelayed(this, 120)
                } else {
                    typingCompleted = true
                    checkAndNavigate()
                }
            }
        }

        handler.post(typingRunnable)
    }

    override fun onResume() {
        super.onResume()
        if (typingCompleted) {
            checkAndNavigate()
        }
    }

    private fun checkAndNavigate() {
        if (hasNavigated) return
        hasNavigated = true

        if (!isInternetAvailable()) {
            Toast.makeText(this, "Working offline. Some features may be limited.", Toast.LENGTH_LONG).show()
        }

        val nextActivity = if (FirebaseAuth.getInstance().currentUser != null) {
            MainActivity::class.java
        } else {
            LoginActivity::class.java
        }
        startActivity(Intent(this, nextActivity))
        finish()
    }

    private fun isInternetAvailable(): Boolean {
        val connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = connectivityManager.activeNetwork ?: return false
        val activeNetwork = connectivityManager.getNetworkCapabilities(network) ?: return false
        return when {
            activeNetwork.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> true
            activeNetwork.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> true
            activeNetwork.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> true
            else -> false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacksAndMessages(null)
    }
}
