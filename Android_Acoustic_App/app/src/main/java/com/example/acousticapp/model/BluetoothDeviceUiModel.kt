package com.example.acousticapp.model

import android.bluetooth.BluetoothDevice

enum class BluetoothConnectionState {
    AVAILABLE,
    CONNECTING,
    CONNECTED,
    DISCONNECTED,
    NOT_CONNECTED
}

data class BluetoothDeviceUiModel(
    val name: String = "",
    val address: String = "",
    val rssi: Int = 0,
    val connectionState: BluetoothConnectionState = BluetoothConnectionState.AVAILABLE,
    val nativeDevice: BluetoothDevice? = null
)
