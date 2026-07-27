package com.example.acousticapp.devices

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.acousticapp.databinding.ItemDeviceBinding
import com.example.acousticapp.model.AcousticDevice

class DeviceAdapter(private var devices: List<AcousticDevice>) : RecyclerView.Adapter<DeviceAdapter.ViewHolder>() {

    class ViewHolder(val binding: ItemDeviceBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemDeviceBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val device = devices[position]
        holder.binding.apply {
            tvDeviceName.text = device.deviceName
            tvDeviceModel.text = "Mode: ${device.mode}"
            tvStatus.text = "${device.connectionStatus} ●"
            tvBattery.text = "${device.batteryStatus}%"
        }
    }

    override fun getItemCount() = devices.size

    fun updateDevices(newDevices: List<AcousticDevice>) {
        devices = newDevices
        notifyDataSetChanged()
    }
}
