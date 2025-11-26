# Generate Datasets using GNU Radio and SDR

This project shows how to generate realistic wideband wireless datasets using **GNU Radio** and an **ADALM–PLUTO SDR**.  
We send and receive real I/Q signals over the air, compute **power spectral density (PSD)** and **SNR**, and then build a large labeled dataset for **modulation recognition**, **spectrum sensing**, and **IoT traffic analysis**. :contentReference[oaicite:0]{index=0}

---

## Main Idea

1. Use an SDR (ADALM–PLUTO) to transmit and receive real signals.
2. Use GNU Radio flowgraphs for:
   - **Primary User (PU) transmitter** – sends modulated symbols.
   - **Secondary User (SU) receiver** – captures I/Q samples and logs PSD/SNR.
3. Save these measurements into `.pth` files.
4. Use `simulate.py` to expand this into a **multi-user, multi-channel** dataset with:
   - 10 primary users (PUs)
   - 10 secondary users (SUs)
   - 20 narrowband channels
   - Multiple modulation types: **BPSK, QPSK, 8PSK, 16QAM** :contentReference[oaicite:1]{index=1}

---

## Repository Contents :contentReference[oaicite:2]{index=2}

- `PU_tx.grc`  
  GNU Radio flowgraph for the **Primary User transmitter**.  
  Sends binary symbols using BPSK/QPSK/8PSK/16QAM via ADALM–PLUTO.

- `SU_rx.grc`  
  GNU Radio flowgraph for the **Secondary User receiver**.  
  Receives I/Q samples from ADALM–PLUTO and sends them to a custom Python block that:
  - computes PSD,
  - estimates SNR,
  - decides if the PU is present,
  - logs data to CSV / `.pth` files. :contentReference[oaicite:3]{index=3}

- `SU_rx_latest.py`  
  Python version of the SU receiver flow (exported/updated from GRC).  
  Use this if you prefer running the receiver as a Python script instead of the GRC GUI.

- `generate_raw_binaries.m`  
  MATLAB script to generate raw binary symbol files (e.g. `symbols_01.bin`) used by the PU transmitter.

- `simulate.py`  
  Script that takes the `.pth` files recorded from the SDR and **builds the final dataset**:
  - creates many PUs and SUs in software,
  - adds path loss, shadowing, frequency drift, and AWGN,
  - outputs a single dataset file ready for training ML models. :contentReference[oaicite:4]{index=4}

---

## Key Concepts (Very Short)

- **I/Q signal** – complex baseband signal:  
  \( x(t) = I(t) + jQ(t) \) where `I` is in-phase, `Q` is quadrature. Used in almost all SDR systems. :contentReference[oaicite:5]{index=5}  
- **Modulation** – how we map bits to a waveform:
  - BPSK, QPSK, 8PSK, 16QAM.
- **PSD (Power Spectral Density)** – shows how signal power is spread over frequency.
- **SNR (Signal-to-Noise Ratio)** – how strong the signal is compared to noise.
- **PU (Primary User)** – “main” transmitter.
- **SU (Secondary User)** – “listener” that senses the spectrum.

---

## Requirements

**Hardware**

- 1 × ADALM–PLUTO SDR (as TX and RX, or as one board with loopback/over-the-air tests). :contentReference[oaicite:6]{index=6}
- PC or laptop with USB.

**Software**

- GNU Radio with GRC (tested with PlutoSDR support).
- ADALM–PLUTO drivers / firmware (RNDIS network interface working).
- Python (with `numpy`, `torch`, etc.).
- MATLAB (optional, for `generate_raw_binaries.m`).

---

## How the Pipeline Works

### 1. Setup SDR and GNU Radio

1. Install GNU Radio on your PC.
2. Install and configure ADALM–PLUTO drivers.
3. Make sure you can **ping** the PlutoSDR over its IP address.
4. Enable PlutoSDR in GNU Radio (Pluto source/sink blocks). :contentReference[oaicite:7]{index=7}

---

### 2. Generate Binary Symbol Files (MATLAB)

1. Open `generate_raw_binaries.m` in MATLAB.
2. Adjust settings if needed (number of symbols, file name, etc.).
3. Run the script – it will create symbol files such as:
   - `symbols_01.bin`

These files are used by `PU_tx.grc` as the input bit stream.

---

### 3. Run the PU Transmitter (PU_tx.grc)

1. Open **`PU_tx.grc`** in GNU Radio Companion.
2. Choose modulation (BPSK, QPSK, 8PSK, or 16QAM) using the constellation block.
3. Set PlutoSDR sink parameters:
   - Center frequency (e.g. 2.4 GHz),
   - Sample rate (e.g. 1.024 MSPS),
   - TX gain, etc. :contentReference[oaicite:8]{index=8}
4. Run the flowgraph to start transmitting.

You can see the transmitted spectrum in the **QT GUI Frequency Sink**.

---

### 4. Run the SU Receiver (SU_rx.grc or SU_rx_latest.py)

1. Open **`SU_rx.grc`** in GNU Radio OR run **`SU_rx_latest.py`**.
2. Set PlutoSDR source parameters to match the transmitter:
   - Same center frequency and sample rate,
   - Proper RX gain and bandwidth. :contentReference[oaicite:9]{index=9}
3. The receiver will:
   - capture complex I/Q samples,
   - send them to the custom PSD logger block,
   - compute PSD and SNR,
   - mark if PU is present or not (simple SNR threshold),
   - store results to CSV and `.pth` files.

At the end you will have **four base `.pth` files**, one per modulation (for example): :contentReference[oaicite:10]{index=10}

- `psd_log_BPSK.pth`
- `psd_log_QPSK.pth`
- `psd_log_8PSK.pth`
- `psd_log_16QAM.pth`

Some additional `.pth` files may group PSDs by SNR bins (2 dB steps) for balanced training.

---

### 5. Build the Final Dataset (simulate.py)

1. Make sure the modulation-specific `.pth` files are in the expected folder.
2. Open `simulate.py` and adjust:
   - input paths,
   - number of classes/samples,
   - SNR list (if needed), etc.
3. Run the script, for example:

   ```bash
   python simulate.py
