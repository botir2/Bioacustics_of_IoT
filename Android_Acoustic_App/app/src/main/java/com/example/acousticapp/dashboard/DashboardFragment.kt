package com.example.acousticapp.dashboard

import android.content.res.ColorStateList
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.navigation.fragment.findNavController
import com.example.acousticapp.R
import com.example.acousticapp.databinding.FragmentDashboardBinding
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

class DashboardFragment : Fragment() {

    private var _binding: FragmentDashboardBinding? = null
    private val binding get() = _binding!!
    private val viewModel: DashboardViewModel by viewModels()
    private val dateFormat = SimpleDateFormat("dd MMM yyyy, HH:mm", Locale.getDefault())
    private val displayDateFormat = SimpleDateFormat("EEEE, dd MMM yyyy", Locale.getDefault())
    private val displayTimeFormat = SimpleDateFormat("HH:mm", Locale.getDefault())

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentDashboardBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        updateRealtimeInfo()
        setupObservers()
        setupListeners()
    }

    private fun updateRealtimeInfo() {
        val now = Calendar.getInstance()
        binding.tvDate.text = displayDateFormat.format(now.time)
        binding.tvTime.text = displayTimeFormat.format(now.time)
        
        val hour = now.get(Calendar.HOUR_OF_DAY)
        val isDaytime = hour in 6..18
        binding.tvDaynightBadge.text = if (isDaytime) "Daytime" else "Nighttime"
    }

    private fun setupObservers() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.uiState.collect { state ->
                    handleUiState(state)
                }
            }
        }
    }

    private fun handleUiState(state: DashboardState) {
        when {
            state.isLoading -> {
                // Show loading if necessary
            }
            state.error != null -> {
                Toast.makeText(requireContext(), state.error, Toast.LENGTH_SHORT).show()
            }
            else -> {
                binding.tvWeather.text = state.weather
                
                if (state.device != null) {
                    val device = state.device
                    
                    // Status Card
                    binding.tvBatteryVal.text = "${device.batteryStatus}%"
                    binding.pbBattery.progress = device.batteryStatus
                    
                    // Connection Formatting
                    binding.tvConnectionVal.text = if (device.connectionStatus.lowercase() in listOf("online", "connected")) getString(R.string.status_on) else getString(R.string.status_off)
                    
                    // Mode Formatting
                    binding.tvModeVal.text = when (device.mode) {
                        "Monitoring" -> getString(R.string.mon_label)
                        "Sleep" -> "Sleep"
                        "Transmit" -> "Tx"
                        "Pre-sense" -> "Pre"
                        else -> device.mode
                    }
                    
                    // Acoustic Status Formatting
                    binding.tvAcousticVal.text = if (device.deviceStatus.lowercase() == "active") "Active" else "Idle"
                    
                    // Latest Detection Card - Image Switching
                    val label = device.lastDetectionLabel.lowercase()
                    val imageRes = when {
                        label.contains("masked owl") -> R.drawable.masked_owl
                        label.contains("cockatoo") -> R.drawable.cockatoo
                        label.contains("sugar glider") -> R.drawable.sugar_glider
                        else -> null
                    }
                    
                    if (imageRes != null) {
                        binding.ivDetectionPlaceholder.setImageResource(imageRes)
                        binding.ivDetectionPlaceholder.imageTintList = null
                        // Use dimension if available, otherwise 0
                        binding.ivDetectionPlaceholder.setPadding(0, 0, 0, 0)
                    } else {
                        binding.ivDetectionPlaceholder.setImageResource(android.R.drawable.ic_menu_gallery)
                        binding.ivDetectionPlaceholder.imageTintList = ColorStateList.valueOf(ContextCompat.getColor(requireContext(), R.color.dash_grey))
                        val padding = (16 * resources.displayMetrics.density).toInt()
                        binding.ivDetectionPlaceholder.setPadding(padding, padding, padding, padding)
                    }
                    
                    binding.tvDetectionEvent.text = device.lastDetectionLabel
                    binding.tvDetectionTime.text = device.lastDetectionTime?.toDate()?.let { dateFormat.format(it) } ?: "N/A"
                }
            }
        }
    }

    private fun setupListeners() {
        binding.btnViewDevices.setOnClickListener { findNavController().navigate(R.id.nav_devices) }
        binding.btnOpenMap.setOnClickListener { findNavController().navigate(R.id.nav_map) }
        binding.btnBluetooth.setOnClickListener { findNavController().navigate(R.id.nav_bluetooth) }
        binding.btnHistory.setOnClickListener { findNavController().navigate(R.id.nav_history) }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
