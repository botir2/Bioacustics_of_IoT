package com.example.acousticapp.history

import androidx.lifecycle.ViewModel
import com.example.acousticapp.model.DetectionEvent
import com.example.acousticapp.repository.DetectionRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

sealed class DetectionHistoryUiState {
    object Loading : DetectionHistoryUiState()
    data class Success(val detections: List<DetectionEvent>) : DetectionHistoryUiState()
    data class Error(val message: String) : DetectionHistoryUiState()
}

class DetectionHistoryViewModel(private val repository: DetectionRepository = DetectionRepository()) : ViewModel() {

    private val _uiState = MutableStateFlow<DetectionHistoryUiState>(DetectionHistoryUiState.Loading)
    val uiState: StateFlow<DetectionHistoryUiState> = _uiState.asStateFlow()

    init {
        loadHistory()
    }

    private fun loadHistory() {
        // TODO: Replace mock data with Firestore data
        val detections = repository.getMockDetections()
        _uiState.value = DetectionHistoryUiState.Success(detections)

    }
}
