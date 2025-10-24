import PySimpleGUI as sg
import numpy as np
from audio.engine import AudioEngine, list_devices
from dsp.processor import Processor
from config.presets import PresetStore
from utils.logger import log

# ------- GUI helpers -------
def build_window(devices):
    in_names = [f"{d['index']}: {d['name']}" for d in devices if d['max_input_channels'] > 0]
    out_names = [f"{d['index']}: {d['name']}" for d in devices if d['max_output_channels'] > 0]

    layout = [
        [sg.Text('Voice Changer Live', font=('Segoe UI', 14, 'bold'))],
        [sg.HorizontalSeparator()],
        [sg.Text('Input device', size=(12,1)), sg.Combo(in_names, key='-IN-', size=(60,1))],
        [sg.Text('Output device', size=(12,1)), sg.Combo(out_names, key='-OUT-', size=(60,1))],
        [sg.Button('Refresh devices', key='-REFRESH-')],
        [sg.HorizontalSeparator()],
        [sg.Text('Pitch (semitones)', size=(16,1)), sg.Slider(range=(-12,12), default_value=0, resolution=0.5, orientation='h', key='-PITCH-', size=(40,15)), sg.Text('0', key='-PVAL-')],
        [sg.Text('Robot amount', size=(16,1)), sg.Slider(range=(0,1), default_value=0.0, resolution=0.05, orientation='h', key='-ROBOT-', size=(40,15)), sg.Text('0', key='-RVAL-')],
        [sg.Text('Reverb amount', size=(16,1)), sg.Slider(range=(0,1), default_value=0.0, resolution=0.05, orientation='h', key='-REVERB-', size=(40,15)), sg.Text('0', key='-REVVAL-')],
        [sg.Checkbox('Enable processing', key='-PROCESS-', default=True), sg.Checkbox('Hear myself (monitor)', key='-MONITOR-', default=True)],
        [sg.HorizontalSeparator()],
        [sg.Button('Start', key='-START-', button_color=('white','#2e7d32')), sg.Button('Stop', key='-STOP-', button_color=('white','#c62828')), sg.Push(),
         sg.Button('Save preset', key='-SAVE-'), sg.Button('Load preset', key='-LOAD-')],
        [sg.StatusBar('Idle', key='-STATUS-')]
    ]
    return sg.Window('Voice Changer Live', layout, finalize=True)


def get_selected_index(combo_val):
    if not combo_val:
        return None
    try:
        return int(str(combo_val).split(':', 1)[0])
    except Exception:
        return None


if __name__ == '__main__':
    sg.theme('DarkBlue3')

    # Presets
    presets = PresetStore('presets.json')

    # DSP Processor (thread-safe config via set_params)
    processor = Processor(sample_rate=44100, block_size=1024)

    # Audio engine (streams)
    devices = list_devices()
    window = build_window(devices)
    engine = AudioEngine(processor=processor)

    # Try select defaults
    try:
        # Choose first available in/out by default
        ins = [x for x in devices if x['max_input_channels']>0]
        outs = [x for x in devices if x['max_output_channels']>0]
        if ins:
            window['-IN-'].update(value=f"{ins[0]['index']}: {ins[0]['name']}")
        if outs:
            window['-OUT-'].update(value=f"{outs[0]['index']}: {outs[0]['name']}")
    except Exception:
        pass

    while True:
        event, values = window.read(timeout=50)
        if event == sg.WIN_CLOSED:
            break

        # Live update labels
        window['-PVAL-'].update(f"{values['-PITCH-']:.1f}")
        window['-RVAL-'].update(f"{values['-ROBOT-']:.2f}")
        window['-REVVAL-'].update(f"{values['-REVERB-']:.2f}")

        # Push params to DSP (engine callback will read atomically)
        processor.set_params(
            enabled=values['-PROCESS-'],
            pitch_semitones=float(values['-PITCH-']),
            robot=float(values['-ROBOT-']),
            reverb=float(values['-REVERB-'])
        )
        engine.set_monitoring(values['-MONITOR-'])

        if event == '-REFRESH-':
            devices = list_devices()
            in_names = [f"{d['index']}: {d['name']}" for d in devices if d['max_input_channels'] > 0]
            out_names = [f"{d['index']}: {d['name']}" for d in devices if d['max_output_channels'] > 0]
            window['-IN-'].update(values=in_names)
            window['-OUT-'].update(values=out_names)
            window['-STATUS-'].update('Devices refreshed')

        elif event == '-START-':
            in_idx = get_selected_index(values['-IN-'])
            out_idx = get_selected_index(values['-OUT-'])
            try:
                engine.start(input_device=in_idx, output_device=out_idx)
                window['-STATUS-'].update('Running…')
            except Exception as e:
                sg.popup_error('Failed to start audio', str(e))
                window['-STATUS-'].update('Error starting')

        elif event == '-STOP-':
            engine.stop()
            window['-STATUS-'].update('Stopped')

        elif event == '-SAVE-':
            name = sg.popup_get_text('Preset name:')
            if name:
                presets.save(name, {
                    'pitch': float(values['-PITCH-']),
                    'robot': float(values['-ROBOT-']),
                    'reverb': float(values['-REVERB-'])
                })
                window['-STATUS-'].update(f"Saved preset '{name}'")

        elif event == '-LOAD-':
            names = presets.list_names()
            if not names:
                sg.popup('No presets saved yet')
            else:
                name = sg.popup_get_text('Type preset name to load:\n' + '\n'.join(names))
                if name and name in names:
                    data = presets.load(name)
                    window['-PITCH-'].update(value=float(data.get('pitch',0)))
                    window['-ROBOT-'].update(value=float(data.get('robot',0)))
                    window['-REVERB-'].update(value=float(data.get('reverb',0)))
                    window['-STATUS-'].update(f"Loaded preset '{name}'")

    engine.stop()
    window.close()