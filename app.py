from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, impulse, step, bode
import os

app = Flask(__name__)

# Path to save generated plots
PLOT_PATH = os.path.join(app.root_path, 'static/plots')

# Ensure the plots directory exists
os.makedirs(PLOT_PATH, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get inputs from the form
            degree_numerator = int(request.form['degree_numerator'])
            numerator = list(map(float, request.form['numerator'].split()))
            degree_denominator = int(request.form['degree_denominator'])
            denominator = list(map(float, request.form['denominator'].split()))
            time_bound = int(request.form['time_bound'])

            # Validate input degrees
            if len(numerator) != (degree_numerator + 1):
                raise ValueError("The number of numerator coefficients must match its degree + 1.")
            if len(denominator) != (degree_denominator + 1):
                raise ValueError("The number of denominator coefficients must match its degree + 1.")
            if len(numerator) >= len(denominator):
                raise ValueError("Numerator degree must be strictly less than Denominator degree.")

            # Generate the transfer function string for display
            numerator_str = " + ".join([f"{num}*s^{i}" for i, num in enumerate(numerator[::-1])])
            denominator_str = " + ".join([f"{den}*s^{i}" for i, den in enumerate(denominator[::-1])])
            transfer_function = f"H(s) = ({numerator_str}) / ({denominator_str})"

            # Create TransferFunction object
            system = TransferFunction(numerator, denominator)

            # Frequency and phase response (Bode plot)
            frequencies = np.logspace(-2, 2, 500)  # Frequency range (0.01 to 100 rad/s)
            w, mag, phase = bode(system, w=frequencies)

            # Time vectors for impulse and step response (from 0 to time_bound)
            time_response = np.linspace(0, time_bound, 1000)
            t_impulse, y_impulse = impulse(system, T=time_response)
            t_step, y_step = step(system, T=time_response)

            # Plot Responses
            plt.figure(figsize=(12, 8))

            # Impulse Response
            plt.subplot(3, 2, 1)
            plt.plot(t_impulse, y_impulse, label="Impulse Response")
            plt.title("Impulse Response")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()

            # Step Response
            plt.subplot(3, 2, 2)
            plt.plot(t_step, y_step, label="Step Response", color='pink')
            plt.title("Step Response")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()

            # Magnitude Response
            plt.subplot(3, 2, 3)
            plt.semilogx(w, mag, label="Magnitude Response")
            plt.title("Frequency Response - Magnitude")
            plt.xlabel("Frequency (rad/s)")
            plt.ylabel("Magnitude (dB)")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()

            # Phase Response
            plt.subplot(3, 2, 4)
            plt.semilogx(w, phase, label="Phase Response", color='orange')
            plt.title("Frequency Response - Phase")
            plt.xlabel("Frequency (rad/s)")
            plt.ylabel("Phase (degrees)")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()

            # Save the plot to a file
            plot_filename = "LTI-PLOTS.png"
            plot_filepath = os.path.join(PLOT_PATH, plot_filename)
            plt.tight_layout()
            plt.savefig(plot_filepath)
            plt.close()

            # Render the results
            return render_template(
                'index.html',
                transfer_function=transfer_function,
                plot_filename=plot_filename
            )

        except ValueError as e:
            return render_template('index.html', error_message=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
