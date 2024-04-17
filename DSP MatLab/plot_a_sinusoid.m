% Parameters
sampling_frequency = 200000;     % Hz
sampling_period = 1 / sampling_frequency;
signal_duration = 5e-3;         % seconds
frequency = 1000;                % 1 kHz
amplitude = 5;

% Time vector
t = 0:sampling_period:signal_duration;

% Sinusoidal signal
sinusoid = amplitude * sin(2 * pi * frequency * t);

% Plotting
figure;
plot(t, sinusoid, 'LineWidth', 2);
title('Sinusoidal Signal'); 
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
