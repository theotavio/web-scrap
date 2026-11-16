import sys
import flet as ft
from src.interface import main_flet


def main():
    if hasattr(ft, "run"):
        ft.run(main_flet)
    elif hasattr(ft, "app"):
        ft.app(target=main_flet)


if __name__ == "__main__":
    main()
