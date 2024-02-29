import customtkinter as ctk
from pathlib import Path
from PIL import Image, ImageTk
from tkVideoPlayer import TkinterVideo

class VideoWindow(ctk.CTk):
    def __init__(self, size=(960, 960)):
        ctk.CTk.__init__(self)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.geometry(f'{size[0]}x{size[1]}')
        self.resizable(False, False)

        self.videoplayer = TkinterVideo(master=self, scaled=True)
        self.videoplayer.load(r"output.avi")
        self.videoplayer.pack(expand=True, fill="both")
        self.videoplayer.play()

        self.media_button = ctk.CTkButton(self, text="Pause", fg_color='gray28', hover_color='#05c46b', corner_radius=0, command=self.toggle_video, width=100, height=38)
        self.media_button.pack(side="left", padx=5, pady=10)

        self.progress_slider = ctk.CTkSlider(master=self, from_=0, to=100, button_color='#485460', button_hover_color='#05c46b', progress_color='gray50')
        self.progress_slider.bind("<Button-1>", self.pause_video)
        self.progress_slider.bind("<ButtonRelease-1>", self.seek)
        self.progress_slider.pack(side="right", fill="x", expand=True, padx=5)

        self.videoplayer.bind("<<Duration>>", self.update_duration)
        self.videoplayer.bind("<<SecondChanged>>", self.update_scale)
        self.videoplayer.bind("<<Ended>>", self.loop)
        self.update_duration()

    def play_video(self):
        self.videoplayer.play()
        self.media_button.configure(text="Pause")

    def pause_video(self, event=None):
        self.videoplayer.pause()
        self.media_button.configure(text="Play")

    def toggle_video(self):
        if self.videoplayer.is_paused():
            self.play_video()
        else:
            self.pause_video()

    def loop(self, event=None):
        self.videoplayer.play()

    def seek(self, event):
        self.videoplayer.seek(int(self.progress_slider.get()))
        self.play_video()

    def update_duration(self, event=None):
        duration = self.videoplayer.video_info()["duration"]
        self.progress_slider["to"] = duration
    
    def update_scale(self, event=None):
        self.update_duration()
        self.progress_slider.set(self.videoplayer.current_duration())

    def video_ended(self, event=None):
        self.progress_slider.set(self.progress_slider["to"])
        self.media_button["text"] = "Play"
        self.progress_slider.set(0)


class ImageWindow(ctk.CTk):
    def __init__(self, size=(960, 960)):
        ctk.CTk.__init__(self)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.geometry(f'{size[0]}x{size[1]}')
        self.resizable(False, False)

        self.image = ctk.CTkLabel(self)
        self.image.pack()

    def setup_image(self, img: ctk.CTkImage):
        self.image.configure(text='', image=ctk.CTkImage(img, size=(960, 960)))
