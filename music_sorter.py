import os
import shutil
from pydub import AudioSegment
import whisper
from pydub.playback import play
from concurrent.futures import ThreadPoolExecutor
import pandas as pd  # Add pandas for table formatting
from tqdm import tqdm  # Add tqdm for progress tracking
import torch  # Add torch for quantization
import platform  # Add platform to check the OS


# Add sound alert function
def sound_alert():
    if platform.system() == "Windows":
        import winsound

        winsound.Beep(1000, 500)  # Frequency 1000 Hz, Duration 500 ms
    else:
        os.system('echo -e "\a"')  # ASCII Bell character


def process_chunk(chunk, model, index, song_name):
    temp_filename = f"temp_chunk_{song_name}_{index}.wav"
    chunk.export(temp_filename, format="wav")
    loaded_audio = whisper.load_audio(temp_filename)
    loaded_audio = whisper.pad_or_trim(loaded_audio)
    mel = whisper.log_mel_spectrogram(loaded_audio, n_mels=model.dims.n_mels).to(
        model.device
    )
    _, probs = model.detect_language(mel)
    os.remove(temp_filename)  # Clean up the temporary file
    return max(probs, key=probs.get)


def identify_language(audio_path, model):
    # Load audio file
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio) // 1000  # duration in seconds
    chunk_size = 30 * 1000  # 30 seconds in milliseconds
    languages = []
    song_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Process audio in chunks without overlap
    chunks = [audio[i : i + chunk_size] for i in range(0, duration * 1000, chunk_size)]

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda index_chunk: process_chunk(
                        index_chunk[1], model, index_chunk[0], song_name
                    ),
                    enumerate(chunks),
                ),
                total=len(chunks),
                desc=f"Processing {song_name}",
            )
        )

    languages.extend(results)

    # Calculate the most frequent language
    if languages:
        return max(set(languages), key=languages.count)
    else:
        return None


def update_csv(queue, csv_path="language_identification_results.csv"):
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path, index_col=0)
        df_new = pd.DataFrame(queue, columns=["Filename", "Language"])
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = pd.DataFrame(queue, columns=["Filename", "Language"])

    df.index += 1  # Start index from 1
    df.to_csv(csv_path, index=True)
    print("\nLanguage identification results updated in CSV file.")


def move_files_from_csv(directory, csv_path="language_identification_results.csv"):
    df = pd.read_csv(csv_path, index_col=0)
    for _, row in tqdm(df.iterrows(), desc="Moving files", total=df.shape[0]):
        fname, lang = row["Filename"], row["Language"]
        language_folder = os.path.join("sorted", lang)
        if not os.path.exists(language_folder):
            os.makedirs(language_folder)
        shutil.move(
            os.path.join(directory, fname), os.path.join(language_folder, fname)
        )


def reverse_file_movement(directory, csv_path="language_identification_results.csv"):
    df = pd.read_csv(csv_path, index_col=0)
    for _, row in tqdm(
        df.iterrows(), desc="Reversing file movement", total=df.shape[0]
    ):
        fname, lang = row["Filename"], row["Language"]
        language_folder = os.path.join("sorted", lang)
        if os.path.exists(os.path.join(language_folder, fname)):
            shutil.move(
                os.path.join(language_folder, fname), os.path.join(directory, fname)
            )


def sort_music_files(directory, move_files=False, reverse_move=False):
    files = os.listdir(directory)
    queue = []

    def process_file(filename):
        file_path = os.path.join(directory, filename)
        language = identify_language(file_path, model_quant)
        torch.cuda.empty_cache()
        return (filename, language)

    processed_files = set()
    csv_path = "language_identification_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        processed_files = set(df["Filename"].tolist())

    files_to_process = [f for f in files if f not in processed_files]

    if reverse_move:
        reverse_file_movement(directory)
    elif move_files:
        move_files_from_csv(directory)
    else:
        # Load Whisper model once with quantization to float16
        model = whisper.load_model("large-v3", download_root="whisper_models")
        model_quant = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.float16
        )
        del model
        torch.cuda.empty_cache()
        for i in range(0, len(files_to_process), 4):
            batch_files = files_to_process[i : i + 4]
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(process_file, batch_files))
                queue.extend(results)

            update_csv(queue)
            sound_alert()
            queue.clear()  # Clear the queue for the next batch
            # break  # Remove this line to process all files


if __name__ == "__main__":
    music_directory = "Music"
    move_files_flag = False  # Set this flag to True to move files as per CSV
    reverse_move_flag = False  # Set this flag to True to reverse file movement
    sort_music_files(
        music_directory, move_files=move_files_flag, reverse_move=reverse_move_flag
    )
