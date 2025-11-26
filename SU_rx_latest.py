import numpy as np
import csv, os, time, random
import torch
from gnuradio import gr

class blk(gr.sync_block):  # one-input, no-output
    def __init__(self):
        gr.sync_block.__init__(self,
            name="psd_logger",
            in_sig=[np.complex64],
            out_sig=[]
        )

        self.fft_size = 1024

        # CSV logging
        self.csv_filename = r"C:\Users\Manish Anwla\Downloads\btpfiles\psd_log_temp.csv"
        if not os.path.exists(self.csv_filename):
            os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
            with open(self.csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Mean_PSD_dB", "SNR_dB", "PU_Present"])

        # Primary PTH logging (original full metadata)
        self.pth_filename = r"C:\Users\Manish Anwla\Downloads\btpfiles\psd_log_temp.pth"
        self.target_psd_len = 192   # change if you want different length (192 matched to MATLAB)
        self.save_every = 5
        self._write_counter = 0

        # Binned-by-SNR PTH (stores pairs (psd, label) per bin)
        self.binned_pth_filename = r"C:\Users\Manish Anwla\Downloads\btpfiles\psd_binned_by_snr_temp.pth"
        self.snr_bin_width = 2.0
        self.snr_min_bin = 2

        # Initialize or load primary pth_data
        if os.path.exists(self.pth_filename):
            try:
                loaded = torch.load(self.pth_filename, map_location='cpu')
                self.pth_data = loaded
                for k in ('timestamps', 'snrs', 'psds', 'mean_psds', 'pu_flags', 'pu_labels'):
                    if k not in self.pth_data:
                        self.pth_data[k] = []
            except Exception as e:
                print("psd_logger: failed to load existing pth file — starting new. Error:", e)
                self.pth_data = {'timestamps': [], 'snrs': [], 'psds': [], 'mean_psds': [], 'pu_flags': [], 'pu_labels': []}
                torch.save(self.pth_data, self.pth_filename)
        else:
            os.makedirs(os.path.dirname(self.pth_filename), exist_ok=True)
            self.pth_data = {'timestamps': [], 'snrs': [], 'psds': [], 'mean_psds': [], 'pu_flags': [], 'pu_labels': []}
            torch.save(self.pth_data, self.pth_filename)

        # Initialize or load binned_data (pairs)
        if os.path.exists(self.binned_pth_filename):
            try:
                loaded2 = torch.load(self.binned_pth_filename, map_location='cpu')
                self.binned_data = loaded2
                # ensure keys
                if 'bins' not in self.binned_data:
                    self.binned_data['bins'] = []
                if 'pairs_by_bin' not in self.binned_data:
                    self.binned_data['pairs_by_bin'] = {}
            except Exception as e:
                print("psd_logger: failed to load existing binned pth — starting new. Error:", e)
                self.binned_data = {'bins': [], 'pairs_by_bin': {}}
                torch.save(self.binned_data, self.binned_pth_filename)
        else:
            os.makedirs(os.path.dirname(self.binned_pth_filename), exist_ok=True)
            self.binned_data = {'bins': [], 'pairs_by_bin': {}}
            torch.save(self.binned_data, self.binned_pth_filename)

    def _resample_psd(self, psd, target_len):
        if target_len is None or target_len == len(psd):
            return psd.copy()
        old_idx = np.arange(len(psd))
        new_idx = np.linspace(0, len(psd)-1, target_len)
        return np.interp(new_idx, old_idx, psd)

    def _snr_to_nearest_bin(self, snr):
        if snr is None or (not np.isfinite(snr)):
            return None
        div = snr / self.snr_bin_width
        low_mult = np.floor(div)
        high_mult = low_mult + 1

        low_bin = int(max(self.snr_min_bin, int(low_mult * self.snr_bin_width)))
        high_bin = int(max(self.snr_min_bin, int(high_mult * self.snr_bin_width)))

        if low_bin == high_bin:
            return low_bin

        dist_low = abs(snr - low_bin)
        dist_high = abs(high_bin - snr)

        if dist_low < dist_high:
            return low_bin
        elif dist_high < dist_low:
            return high_bin
        else:
            return random.choice([low_bin, high_bin])

    def _ensure_bin_exists(self, bin_center):
        bc = int(bin_center)
        if bc not in self.binned_data['bins']:
            self.binned_data['bins'].append(bc)
            self.binned_data['bins'] = sorted(self.binned_data['bins'])
        if bc not in self.binned_data['pairs_by_bin']:
            self.binned_data['pairs_by_bin'][bc] = []

    def work(self, input_items, output_items):
        x = input_items[0]
        if x is None or x.size == 0:
            return 0

        # pick analysis window (use first target_psd_len window length or smaller window then nfft)
        # Here we use the convention: compute PSD from window length = target_psd_len//3 if you want MATLAB-style 64->192
        # But since you set target_psd_len=192 and want window_len=64, compute from 64 and FFT with n=192:
        analysis_win_len = 64
        nfft = 3 * analysis_win_len  # 192

        if x.size < analysis_win_len:
            win = np.zeros(analysis_win_len, dtype=np.complex64)
            win[:x.size] = x
        else:
            win = x[:analysis_win_len]

        psd_raw = np.abs(np.fft.fftshift(np.fft.fft(win, n=nfft)))**2
        psd_dB = 10.0 * np.log10(psd_raw + 1e-12)   # shape (nfft,)

        mean_psd = float(np.mean(psd_dB))
        noise_floor = float(np.percentile(psd_dB, 10))
        snr = mean_psd - noise_floor
        pu_flag = 1 if snr > 5 else 0

        # CSV logging
        try:
            with open(self.csv_filename, "a", newline="") as f:
                csv.writer(f).writerow([time.time(), mean_psd, snr, pu_flag])
        except Exception as e:
            print("psd_logger: failed to write CSV:", e)

        # prepare PSD to save: resample to target_psd_len (if set), and keep shape (L,1)
        psd_for_save = self._resample_psd(psd_dB, self.target_psd_len)
        psd_for_save = psd_for_save.reshape(self.target_psd_len, 1)

        # ---------- primary pth ----------
        self.pth_data['timestamps'].append(float(time.time()))
        self.pth_data['snrs'].append(float(snr))
        self.pth_data['mean_psds'].append(float(mean_psd))
        self.pth_data['pu_flags'].append(int(pu_flag))
        self.pth_data['pu_labels'].append(int(pu_flag))
        self.pth_data['psds'].append(psd_for_save.copy())

        # ---------- binned pth storing pairs (psd,label) ----------
        bin_center = self._snr_to_nearest_bin(snr)
        if bin_center is not None:
            self._ensure_bin_exists(bin_center)
            # append pair (psd, label) to that bin
            # Note: store psd as numpy array, label as int
            self.binned_data['pairs_by_bin'][bin_center].append( (psd_for_save.copy(), int(pu_flag)) )

        # Save periodically
        self._write_counter += 1
        if self._write_counter >= self.save_every:
            try:
                torch.save(self.pth_data, self.pth_filename)
            except Exception as e:
                print("psd_logger: failed to write pth:", e)
            try:
                torch.save(self.binned_data, self.binned_pth_filename)
            except Exception as e:
                print("psd_logger: failed to write binned pth:", e)
            self._write_counter = 0

        return len(input_items[0])
