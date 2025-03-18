import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from settings import states, quality_ofc, quality_emg, ofc_lims, emg_lims  

from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox

# Import your Epochs class (adjust the import path as needed)
from epochs import Epochs


class EEGVideoApp:
    def __init__(self):
        self.video_path = None
        self.epochs_obj = None
        self.eeg_data = None
        self.metadata = None
        self.feats = None
        self.animal_id = None
        self.current_epoch_index = None
        self.frame_start = None
        self.frame_end = None
        self.current_label = 'None'
        self.all_labels = None
        self.cap = None
        self.eeg_plot_image = None  # Cache for the combined plot image
        
        # Create a figure with 1 row and 3 columns:
        # Column 1: EEG plot; Column 2: EMG plot; Column 3: Features barplot.
        self.fig, self.axs = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.canvas.draw()

        # Make sure that the keys in states don't interfere with reserved keys
        assert not any([key in ['q', ']', '[', ' ', '+', '-', 's'] for key in states.keys()]), "States keys interfere with keypresses!"

        # Load the data
        self.load_app()

    def load_app(self):
        self.load_video()
        self.load_epochs_data()
        self.load_labels()

    def load_video(self):
        video_file_path, _ = QFileDialog.getOpenFileName(
            None, "Open Video File", "", "Video Files (*.mp4 *.avi)"
        )
        if video_file_path:
            self.cap = cv2.VideoCapture(video_file_path)

    def load_epochs_data(self):
        # Ask for the Epochs object file (e.g. a gzip file saved via Epochs.save_gz)
        epochs_file, _ = QFileDialog.getOpenFileName(
            None, "Open Epochs Object File", "", "GZip Files (*.gz)"
        )
        if epochs_file:
            self.epochs_obj = Epochs.load_gz(epochs_file)
            # Reset metadata index to ensure proper indexing for epochs
            self.metadata = self.epochs_obj.metadata.reset_index()
            # Get animal_id from metadata (assumes a column 'animal_id' exists)
            self.animal_id = int(self.metadata['animal_id'].unique()[0])
            # Extract EEG and EMG data using your quality settings.
            # (If quality_ofc returns 'Both', use "OFC_L" by default.)
            self.eeg_data = self.epochs_obj.epochs.get_data(
                picks=[
                    quality_ofc[self.animal_id] if quality_ofc[self.animal_id] != 'Both' else "OFC_L",
                    quality_emg[self.animal_id]
                ]
            )
            # Also extract the epoch feature values (assumes self.epochs_obj.feats is a DataFrame)
            self.feats = self.epochs_obj.feats
            self.current_epoch_index = 0
            self.get_epoch_frames()
            self.update_eeg_plot()

    def load_labels(self):
        labels_file, _ = QFileDialog.getOpenFileName(
            None,
            "Open Labels File (Click 'Cancel' if you don't wish to load past labelling)",
            "",
            "Excel Files (*.xlsx)"
        )
        if labels_file:
            self.all_labels = pd.read_excel(labels_file)
        else:
            print("No past labels were used")
            # Create a labels DataFrame from metadata with an added 'Label' column
            self.all_labels = self.metadata.copy()
            self.all_labels['Label'] = None

    def get_epoch_frames(self):
        # Use metadata columns 'start_frame' and 'end_frame' to set video boundaries for the epoch
        frame_start = self.metadata.loc[self.current_epoch_index, 'start_frame']
        frame_end = self.metadata.loc[self.current_epoch_index, 'end_frame']
        if pd.isna(frame_start) or pd.isna(frame_end):
            self.frame_start = 0
            self.frame_end = 1
        else:
            self.frame_start = int(frame_start)
            self.frame_end = int(frame_end)

    def update_eeg_plot(self):
        # Clear all three axes
        for ax in self.axs:
            ax.clear()

        # --- EEG Plot (Column 1) ---
        self.axs[0].plot(self.eeg_data[self.current_epoch_index][0], label="EEG", color="black")
        self.axs[0].set_title("EEG Signal")
        self.axs[0].set_ylabel("Amplitude")
        self.axs[0].set_ylim(ofc_lims)

        # --- EMG Plot (Column 2) ---
        self.axs[1].plot(self.eeg_data[self.current_epoch_index][1], label="EMG", color="red")
        self.axs[1].set_title("EMG Signal")
        self.axs[1].set_ylabel("Amplitude")
        self.axs[1].set_xlabel("Time")
        self.axs[1].set_ylim(emg_lims)

        # --- Features Barplot (Column 3) ---
        # Get the feature values for the current epoch (assumes self.feats is a DataFrame)
        feature_row = self.feats.iloc[self.current_epoch_index]
        feature_names = feature_row.index.tolist()
        feature_values = feature_row.values
        self.axs[2].barh(feature_names, feature_values, color="green")
        self.axs[2].set_title("Features")
        self.axs[2].set_xlabel("Value")

        # Adjust layout and draw the canvas
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plot = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        plot = plot.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.eeg_plot_image = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)

    def update_frame(self, reverse=False):
        if reverse:
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) > self.frame_start:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 2)
            else:
                return

        ret, frame = self.cap.read()

        # Loop video if we reach the end of the current epoch’s frames
        if not ret or self.cap.get(cv2.CAP_PROP_POS_FRAMES) > self.frame_end:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start)
            return

        # Add epoch number, current frame, and label text onto the video frame
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        text = f"Epoch: {self.current_epoch_index + 1} | Frame: {current_frame} | Label: {self.current_label}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Resize the combined EEG/EMG/Features plot image to roughly match the video frame height
        height = frame.shape[0]
        scale = height / self.eeg_plot_image.shape[0]
        new_width = int(self.eeg_plot_image.shape[1] * scale)
        eeg_plot_resized = cv2.resize(self.eeg_plot_image, (new_width, height))

        # Horizontally stack the video frame and the plot image
        combined = np.hstack([frame, eeg_plot_resized])
        cv2.imshow("EEG and Video App", combined)

    def label_epoch(self):
        self.all_labels.loc[self.current_epoch_index, 'Label'] = self.current_label

    def save_labels(self):
        save_path, _ = QFileDialog.getSaveFileName(
            None, "Save Labels", "", "Excel Files (*.xlsx)"
        )
        if save_path:
            self.all_labels.to_excel(save_path, index=False)
        else:
            print("No save path selected.")

    def update_label(self):
        self.current_label = self.all_labels.loc[self.current_epoch_index, 'Label']

    def confirm_quit(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Quit Confirmation")
        msg_box.setText("Are you sure you want to quit?")
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        response = msg_box.exec_()
        return response == QMessageBox.Yes

    def display_epoch(self):
        self.get_epoch_frames()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start)
        paused = False
        while True:
            if not paused:
                self.update_frame()

            # Check for keypress events
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):  # Quit
                if self.confirm_quit():
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break
            elif key == ord(']'):  # Next epoch
                self.next_epoch()
                break
            elif key == ord('['):  # Previous epoch
                self.prev_epoch()
                break
            elif key == ord(' '):  # Pause/unpause
                paused = not paused
            elif key == ord('+'):
                self.update_frame()
            elif key == ord('-'):
                self.update_frame(reverse=True)
            elif key == 6:
                self.find_epoch()
            # Label the epoch based on keypress from the "states" dictionary
            elif chr(key) in states.keys():
                self.current_label = states[chr(key)]
                print(f"Epoch: {self.current_epoch_index} - Label: {self.current_label}")
                self.label_epoch()
                self.update_frame()
            elif key == ord('s'):
                self.save_labels()
                print('Labels saved to excel file.')
        self.save_labels()
        sys.exit()

    def next_epoch(self):
        if self.current_epoch_index < len(self.metadata) - 1:
            self.current_epoch_index += 1
            self.get_epoch_frames()
            self.update_eeg_plot()
            self.update_label()
            self.display_epoch()

    def prev_epoch(self):
        if self.current_epoch_index > 0:
            self.current_epoch_index -= 1
            self.get_epoch_frames()
            self.update_eeg_plot()
            self.update_label()
            self.display_epoch()

    def find_epoch(self):
        epoch_number, ok = QInputDialog.getInt(None, "Find Epoch", "Enter Epoch Number:")
        if ok:
            if 1 <= epoch_number <= len(self.eeg_data):
                self.current_epoch_index = epoch_number - 1
                self.get_epoch_frames()
                self.update_eeg_plot()
                self.update_label()
                self.display_epoch()
            else:
                QMessageBox.warning(None, "Invalid Epoch", "Invalid epoch number. Please enter a valid epoch number.")


if __name__ == "__main__":
    app = EEGVideoApp()
    app.display_epoch()
    sys.exit()
