# Firestore Map Schema Draft

## Collections

### `users/{uid}`
Existing user profile data.

### `devices/{deviceId}`
Store state and location of each physical device.
- `ownerUid`: String (UID of the user who owns this device)
- `deviceName`: String
- `latitude`: Double
- `longitude`: Double
- `gpsAddress`: String
- `acousticStatus`: String (e.g., "Normal", "Alert")
- `connectionStatus`: String (e.g., "Online", "Offline")
- `batteryLevel`: Int (0-100)
- `deviceMode`: String (e.g., "ACTIVE", "SLEEP")
- `lastSeen`: Timestamp (Server Timestamp)

### `detections/{detectedId}`
History of detections triggered by devices.
- `uid`: String (UID of the owner)
- `deviceId`: String
- `detectedClass`: String (e.g., "SoundTypeA", "SoundTypeB")
- `confidence`: Double (0.0 to 1.0)
- `detectedTime`: Timestamp
- `acousticStatus`: String
- `connectionStatus`: String
- `batteryLevel`: Int
- `deviceMode`: String
- `latitude`: Double
- `longitude`: Double
- `gpsAddress`: String
