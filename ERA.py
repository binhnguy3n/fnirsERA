import streamlit as st
import mne
import tempfile
import os
import matplotlib.pyplot as plt

# Force MNE to use matplotlib for Streamlit compatibility (headless rendering)
mne.viz.set_browser_backend('matplotlib')

st.set_page_config(page_title="fNIRS ERA Viewer", layout="wide")
st.title("fNIRS Event-Related Averages Viewer")

uploaded_file = st.file_uploader("Upload your NIRx .snirf file", type=['snirf'])

if uploaded_file is not None:
    # MNE requires a file path, so we save the uploaded BytesIO object to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".snirf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner("Loading SNIRF data..."):
            # Load raw data
            raw = mne.io.read_raw_snirf(tmp_path, preload=True)
            
        # Determine data type and convert to Concentration Changes (CC) if needed
        st.subheader("Data Preprocessing")
        channel_types = raw.get_channel_types()
        
        with st.spinner("Converting to Concentration Changes..."):
            if 'fnirs_cw_amplitude' in channel_types:
                raw_od = mne.preprocessing.nirs.optical_density(raw)
                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
                st.success("Converted Raw Amplitude -> Optical Density -> Concentration Changes")
            elif 'fnirs_od' in channel_types:
                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw)
                st.success("Converted Optical Density -> Concentration Changes")
            elif 'hbo' in channel_types or 'hbr' in channel_types:
                raw_haemo = raw
                st.success("Data is already in Concentration Changes (HbO/HbR)")
            else:
                st.error("Unrecognized fNIRS channel types.")
                st.stop()

        # 1. Plot Continuous Data (CC)
        st.subheader("Original Data (Concentration Changes)")
        # Plot a subset of channels to prevent the browser from crashing on large datasets
        fig_raw = raw_haemo.plot(n_channels=20, duration=100, show=False)
        st.pyplot(fig_raw)

        # 2. Extract Events & Epochs
        st.subheader("Event-Related Averages (ERA)")
        events, event_dict = mne.events_from_annotations(raw_haemo)

        if len(events) == 0:
            st.warning("No events or annotations found in the SNIRF file.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                tmin = st.number_input("Epoch Start (tmin in seconds)", value=-5.0, step=1.0)
            with col2:
                tmax = st.number_input("Epoch End (tmax in seconds)", value=15.0, step=1.0)

            # Create Epochs
            epochs = mne.Epochs(
                raw_haemo, 
                events, 
                event_id=event_dict,
                tmin=tmin, 
                tmax=tmax,
                baseline=(None, 0), 
                preload=True,
                verbose=False
            )

            # Plot Evoked Responses for each condition
            for event_name in event_dict.keys():
                st.markdown(f"**Condition: {event_name}**")
                evoked = epochs[event_name].average()
                
                # MNE plotting for evoked data
                fig_evoked = evoked.plot(show=False)
                st.pyplot(fig_evoked)

    except Exception as e:
        st.error(f"Error processing the file: {e}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
