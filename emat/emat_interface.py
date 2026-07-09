"""Thin context-managed wrapper around the Vitesse EMAT API."""

from VitesseAPI import Vitesse
import numpy as np
import numpy.typing
from scipy import signal


class EMATSession:
    """Manage EMAT connection lifetime and acquisition configuration."""

    def __init__(self):
        """Initialize with no active Vitesse session."""
        self.V = None

    def __enter__(self):
        """Open the underlying API session and return this wrapper."""
        self.V = Vitesse().initialise().__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Close the underlying API session if it was opened."""
        if self.V is not None:
            self.V.__exit__(exc_type, exc, tb)

    def configure(self):
        """Apply scan settings tuned for current EMAT workflow."""
        self.V.setConfig(
            opFrequency=3.6e6,
            numCycles=2,
            channelsOnDrive=[1,0,0,0,0,0,0,0],
            channelsOnReceive=[1,0,0,0,0,0,0,0],
            numAverages=1000,
            PRF=1000,
            recordLength=50e-6
            #Edit the above parameters to fine tune data acquisition.
        )

    def acquire(self):
        """Return the latest waveform array from the EMAT device."""
        samplingFrequency = 50e6
        numCycles = 2
        signalFrequency = 3.6e6
        nyquistFrequency = 0.5 * samplingFrequency  # nyquist frequency
        # frequency we want to ignore below
        lowcut = (1-(1/numCycles))*signalFrequency
        # frequency we want to ignore above
        highcut = (1+(1/numCycles))*signalFrequency
        low = lowcut/nyquistFrequency
        high = highcut/nyquistFrequency
    
        # bandpass filter applied to raw signal from amplifier
        b, a = signal.butter(2, [low, high], btype='band')
    
        filtAscan = signal.filtfilt(b, a, self.V.getArray())
        # hilbert filter applied to filtered signal from amplifier
        hilbAscan = abs(
            signal.hilbert(filtAscan))
        return filtAscan