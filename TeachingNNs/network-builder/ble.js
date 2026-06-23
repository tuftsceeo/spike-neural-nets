const SERVICE_UUID = '0000fd02-0000-1000-8000-00805f9b34fb';
const WRITE_UUID   = '0000fd02-0001-1000-8000-00805f9b34fb';
const NOTIFY_UUID  = '0000fd02-0002-1000-8000-00805f9b34fb';

export class BLEDevice {
    constructor() {
        this.device = null;
        this.server = null;
        this.service = null;
        this.writeCharacteristic = null;
        this.notifyCharacteristic = null;
        this.callback = null;
    }

    async scan() {
        this.device = await navigator.bluetooth.requestDevice({
            filters: [{ services: [SERVICE_UUID] }]
        });
    }

    async connect(mycallback) {
        // Store the Python notification callback (or null)
        this.callback = mycallback;

        if (!this.device) {
            console.error('No device selected. Call scan() first.');
            return;
        }

        try {
            this.server = await this.device.gatt.connect();
            this.service = await this.server.getPrimaryService(SERVICE_UUID);
            this.writeCharacteristic = await this.service.getCharacteristic(WRITE_UUID);
            this.notifyCharacteristic = await this.service.getCharacteristic(NOTIFY_UUID);

            // Bind so 'this' is correct inside the handler
            this.handleNotification = this.handleNotification.bind(this);
            await this.notifyCharacteristic.startNotifications();
            this.notifyCharacteristic.addEventListener(
                'characteristicvaluechanged',
                this.handleNotification
            );

            console.log('BLE connected to:', this.device.name);
        } catch (error) {
            console.error('Error connecting to BLE device:', error);
            throw error; // Re-throw so Python's try/except can catch it
        }
    }

    handleNotification(event) {
        const value = event.target.value;
        const data = new Uint8Array(value.buffer);

        if (this.callback) {
            try {
                // Pass NOTIFY_UUID and the data array to the Python callback.
                // The Python side converts data with data.to_py() -> bytes().
                this.callback(NOTIFY_UUID, data);
            } catch (err) {
                console.error('Error in BLE notification callback:', err);
            }
        }
    }

    async send(data) {
        if (!this.writeCharacteristic) {
            console.error('Not connected — cannot send data.');
            return;
        }
        try {
            const bytes = new Uint8Array(data);
            await this.writeCharacteristic.writeValue(bytes);
        } catch (error) {
            console.error('Error writing to BLE device:', error);
            throw error;
        }
    }

    disconnect() {
        if (this.device && this.device.gatt.connected) {
            this.device.gatt.disconnect();
            console.log('BLE disconnected.');
        }
    }
}