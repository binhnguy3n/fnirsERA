import os
import tempfile
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import sem
import streamlit as st

# Force MNE to use matplotlib for headless rendering in Streamlit Cloud
mne.viz.set_browser_backend("matplotlib")

st.set_page_config(page_title="fNIRS ERA Viewer", layout="wide")
st.title("fNIRS Event-Related Averages (ERA) Viewer")

uploaded_file = st.file_uploader("Upload your NIRx .snirf file", type=["snirf"])

if uploaded_file is not None:
    # Save uploaded bytes to a temporary file because MNE requires a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".snirf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner("Loading SNIRF data..."):
            raw = mne.io.read_raw_snirf(tmp_path, preload=True)

        st.subheader("1. Preprocessing & Concentration Conversion")
        channel_types = raw.get_channel_types()

        with st.spinner("Converting to Concentration Changes (HbO / HbR)..."):
            if "fnirs_cw_amplitude" in channel_types:
                raw_od = mne.preprocessing.nirs.optical_density(raw)
                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
                st.success("Converted: Raw Amplitude → Optical Density → Concentration Changes")
            elif "fnirs_od" in channel_types:
                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw)
                st.success("Converted: Optical Density → Concentration Changes")
            elif "hbo" in channel_types or "hbr" in channel_types:
                raw_haemo = raw
                st.success("Data is already formatted as Concentration Changes (HbO/HbR).")
            else:
                st.error("Unrecognized fNIRS channel types in file.")
                st.stop()

        # Display continuous data plot
        st.subheader("2. Continuous Data (Concentration Changes)")
        with st.expander("Show Continuous Data Plot", expanded=False):
            fig_raw = raw_haemo.plot(n_channels=20, duration=60, show=False)
            st.pyplot(fig_raw)
            plt.close(fig_raw)

        # Event extraction & Epoching
        st.subheader("3. Event-Related Averages (ERA)")
        events, event_dict = mne.events_from_annotations(raw_haemo)

        if len(events) == 0:
            st.warning("No events or annotations found in the SNIRF file.")
        else:
            # Epoching controls
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                tmin = st.number_input("Epoch Start (tmin in seconds)", value=-5.0, step=1.0)
            with col_t2:
                tmax = st.number_input("Epoch End (tmax in seconds)", value=30.0, step=1.0)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=tmin,
                tmax=tmax,
                baseline=(None, 0),
                preload=True,
                verbose=False,
            )

            # Identify source-detector pairs
            sd_pairs = sorted(
                list(
                    set(
                        [
                            ch.rsplit(" ", 1)[0]
                            for ch in epochs.ch_names
                            if "hbo" in ch.lower()
                        ]
                    )
                )
            )

            if not sd_pairs:
                st.error("Could not find matching HbO/HbR channel pairs.")
                st.stop()

            # Selection widgets
            col_cond, col_chan = st.columns(2)
            with col_cond:
                selected_event = st.selectbox("Select Condition", list(event_dict.keys()))
            with col_chan:
                selected_sd = st.selectbox("Select Source-Detector Channel", sd_pairs)

            # Channel names in MNE
            ch_hbo = f"{selected_sd} hbo"
            ch_hbr = f"{selected_sd} hbr"

            if ch_hbo in epochs.ch_names and ch_hbr in epochs.ch_names:
                # Extract epoch data (scaled from M to µM)
                data_hbo = epochs[selected_event].get_data(picks=ch_hbo)[:, 0, :] * 1e6
                data_hbr = epochs[selected_event].get_data(picks=ch_hbr)[:, 0, :] * 1e6
                times = epochs.times

                # Calculate Mean and Standard Error (SEM)
                mean_hbo, sem_hbo = np.mean(data_hbo, axis=0), sem(data_hbo, axis=0)
                mean_hbr, sem_hbr = np.mean(data_hbr, axis=0), sem(data_hbr, axis=0)

                # Figure styling matching the paper layout
                fig, ax = plt.subplots(figsize=(6, 3.8), dpi=120)

                # Plot HbO (Red)
                ax.plot(times, mean_hbo, color="#D9381E", label="HbO", linewidth=2.2)
                ax.fill_between(
                    times,
                    mean_hbo - sem_hbo,
                    mean_hbo + sem_hbo,
                    color="#D9381E",
                    alpha=0.18,
                )

                # Plot HbR (Blue)
                ax.plot(times, mean_hbr, color="#0080C0", label="HbR", linewidth=2.2)
                ax.fill_between(
                    times,
                    mean_hbr - sem_hbr,
                    mean_hbr + sem_hbr,
                    color="#0080C0",
                    alpha=0.18,
                )

                # Axes styling
                display_title = selected_sd.replace("_", "-")
                ax.set_title(display_title, loc="left", fontsize=11, fontweight="bold")
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.set_xlabel("Time (s)", fontsize=10)
                ax.set_ylabel("µM", fontsize=10)

                # Remove top and right spines
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Match clean legend style
                ax.legend(frameon=False, loc="upper right", fontsize=9)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning(f"Could not find both HbO and HbR traces for {selected_sd}.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
