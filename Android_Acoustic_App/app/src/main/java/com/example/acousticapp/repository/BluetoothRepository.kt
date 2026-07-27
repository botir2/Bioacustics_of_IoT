package com.example.acousticapp.repository

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.Context
import com.example.acousticapp.model.BluetoothConnectionState
import com.example.acousticapp.model.BluetoothDeviceUiModel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

class BluetoothRepository(private val context: Context) {
    private val bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    private val bluetoothAdapter: BluetoothAdapter? = bluetoothManager.adapter
    private val scanner = bluetoothAdapter?.bluetoothLeScanner

    private var activeGatt: BluetoothGatt? = null

    @SuppressLint("MissingPermission")
    fun startScan(): Flow<BluetoothDeviceUiModel> = callbackFlow {
        val scanCallback = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                val device = result.device
                val name = device.name ?: "Unknown BLE Device"
                val address = device.address
                val rssi = result.rssi
                
                trySend(BluetoothDeviceUiModel(
                    name = name,
                    address = address,
                    rssi = rssi,
                    connectionState = BluetoothConnectionState.AVAILABLE,
                    nativeDevice = device
                ))
            }

            override fun onScanFailed(errorCode: Int) {
                close()
            }
        }

        scanner?.startScan(scanCallback)
        
        awaitClose {
            scanner?.stopScan(scanCallback)
        }
    }

    @SuppressLint("MissingPermission")
    fun connectToDevice(
        device: BluetoothDevice,
        onStateChange: (BluetoothConnectionState) -> Unit
    ) {
        activeGatt?.disconnect()
        activeGatt?.close()

        activeGatt = device.connectGatt(context, false, object : BluetoothGattCallback() {
            override fun onConnectionStateChange(gatt: BluetoothGatt?, status: Int, newState: Int) {
                when (newState) {
                    BluetoothProfile.STATE_CONNECTED -> {
                        onStateChange(BluetoothConnectionState.CONNECTED)
                    }
                    BluetoothProfile.STATE_DISCONNECTED -> {
                        onStateChange(BluetoothConnectionState.DISCONNECTED)
                        gatt?.close()
                    }
                    BluetoothProfile.STATE_CONNECTING -> {
                        onStateChange(BluetoothConnectionState.CONNECTING)
                    }
                }
            }
        })
    }

    fun isBluetoothEnabled(): Boolean {
        return bluetoothAdapter?.isEnabled == true
    }
}
