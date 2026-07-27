# Future Firebase, Bluetooth, and Google Maps Integration TODO List

## Authentication
- [ ] TODO: Replace mock login status check with real `FirebaseAuth.getInstance().currentUser`.
- [ ] TODO: Implement Google Sign-In button logic in `LoginActivity`.

## Dashboard
- [ ] TODO: Replace `getMockDevices()` with a Firestore snapshot listener on the `devices` collection (filtered by `ownerUid`).
- [ ] TODO: Replace `getMockDetections()` with a Firestore query on the `detections` collection.
- [ ] TODO: Calculate real-time battery average from Firestore data.

## GPS Map
- [ ] TODO: Integrate Google Maps SDK (Add API Key in `AndroidManifest.xml`).
- [ ] TODO: Initialize `GoogleMap` in `MapFragment`.
- [ ] TODO: Plot markers using `DeviceLocation` data fetched from Firestore.
- [ ] TODO: Implement `onMarkerClick` to show device detail cards.

## My Devices
- [ ] TODO: Implement `rv_devices` adapter to bind Firestore `AcousticDevice` documents.
- [ ] TODO: Implement "Add Device" FAB logic to save new device info to Firestore.

## Bluetooth
- [ ] TODO: Replace `getMockBluetoothDevices()` with real BLE scan results using `BluetoothLeScanner`.
- [ ] TODO: Implement `startScan()` with proper runtime permission checks for `BLUETOOTH_SCAN`.
- [ ] TODO: Handle `BLUETOOTH_CONNECT` permission before connecting to a device.
- [ ] TODO: Implement `BluetoothGattCallback` in `BluetoothDeviceDetailFragment` to read live acoustic data.

## Detection History
- [ ] TODO: Implement pagination or infinite scroll for the `detections` list from Firestore.
- [ ] TODO: Add filter logic (by device, by event type).

## Profile
- [ ] TODO: Fetch real user profile data from `users` Firestore collection.
- [ ] TODO: Implement Profile Picture upload to Firebase Storage.

## General
- [ ] TODO: Implement Offline Support using Firestore Persistence.
- [ ] TODO: Setup Firebase Cloud Messaging (FCM) for "Alert" detections.
