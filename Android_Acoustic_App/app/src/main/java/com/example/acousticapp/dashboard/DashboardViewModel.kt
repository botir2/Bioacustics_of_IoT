package com.example.acousticapp.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.acousticapp.model.AcousticDevice
import com.example.acousticapp.repository.DeviceRepository
import com.example.acousticapp.repository.WeatherRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class DashboardState(
    val device: AcousticDevice? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    val noData: Boolean = false,
    val weather: String = "Loading weather..."
)

class DashboardViewModel(
    private val deviceRepo: DeviceRepository = DeviceRepository(),
    private val weatherRepo: WeatherRepository = WeatherRepository()
) : ViewModel() {

    private val _uiState = MutableStateFlow(DashboardState())
    val uiState: StateFlow<DashboardState> = _uiState.asStateFlow()

    init {
        observeDeviceData()
        fetchWeather()
    }

    private fun fetchWeather() {
        viewModelScope.launch {
            val weatherInfo = weatherRepo.getHobartWeather()
            _uiState.value = _uiState.value.copy(weather = weatherInfo)
        }
    }

    private fun observeDeviceData() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null, noData = false)
            deviceRepo.observeDevice("device_001").collect { result ->
                result.fold(
                    onSuccess = { device ->
                        if (device != null) {
                            _uiState.value = _uiState.value.copy(device = device, isLoading = false)
                        } else {
                            _uiState.value = _uiState.value.copy(isLoading = false, noData = true)
                        }
                    },
                    onFailure = { throwable ->
                        _uiState.value = _uiState.value.copy(isLoading = false, error = throwable.message)
                    }
                )
            }
        }
    }
}
