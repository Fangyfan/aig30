from typing import Literal, IO
from colorama import Fore, Back, Style
from typing import Literal


def color_print(*text, **keyword):
    text = ' '.join([f'{keyword["tips"]}{keyword["division"]}', *text])
    text = keyword['color'] + text + Style.RESET_ALL
    end = keyword['end'] if 'end' in keyword else '\n'
    end = keyword['color'] + end + Style.RESET_ALL
    print(text, sep=keyword['sep'], end=end,
          file=keyword['file'], flush=keyword['flush'])


class Log:
    _show_print = True
    _show_error = True
    _show_success = True
    _show_warning = True
    _show_note = True
    _show_tips = False

    def show(prin=True, error=True, success=True, warnin=True, note=True, tips=True, all=False):
        '''
        默认显示所有
        '''
        if all:
            Log._show_print = True
            Log._show_error = True
            Log._show_success = True
            Log._show_warning = True
            Log._show_note = True
            return
        if prin:
            Log._show_print = True
        if error:
            Log._show_error = True
        if success:
            Log._show_success = True
        if warnin:
            Log._show_warning = True
        if note:
            Log._show_note = True
        if tips:
            Log._show_tips = True

    def hide(prin=True, error=False, success=True, warnin=False, note=True, tips=True, all=False):
        '''
        默认不关闭error和warning的显示
        '''
        if all:
            Log._show_print = False
            Log._show_error = False
            Log._show_success = False
            Log._show_warning = False
            Log._show_note = False
            return
        if prin:
            Log._show_print = False
        if error:
            Log._show_error = False
        if success:
            Log._show_success = False
        if warnin:
            Log._show_warning = False
        if note:
            Log._show_note = False
        if tips:
            Log._show_tips = False

    def print(*values: object,
              sep: str | None = " ",
              end: str | None = "\n",
              file: IO | None = None,
              flush: Literal[False] = False,
              ):
        if Log._show_print:
            print(*values, sep=sep, end=end, file=file, flush=flush)

    def error(*values: object,
              sep: str | None = " ",
              end: str | None = "\n",
              file: IO | None = None,
              flush: Literal[False] = False,
              tips: str = '错误',
              division=':',
              color=Fore.RED,
              ):
        if Log._show_error:
            color_print(*values, sep=sep, end=end, file=file,
                        flush=flush, tips=tips, color=color, division=division)

    def success(*values: object,
                sep: str | None = " ",
                end: str | None = "\n",
                file: IO | None = None,
                flush: Literal[False] = False,
                tips: str = '成功',
                division=':',
                color=Fore.GREEN,
                ):
        if Log._show_success:
            color_print(*values, sep=sep, end=end, file=file,
                        flush=flush, tips=tips, color=color, division=division)

    def warning(*values: object,
                sep: str | None = " ",
                end: str | None = "\n",
                file: IO | None = None,
                flush: Literal[False] = False,
                tips: str = '警告',
                division=':',
                color=Fore.YELLOW,
                ):
        if Log._show_warning:
            color_print(*values, sep=sep, end=end, file=file,
                        flush=flush, tips=tips, color=color, division=division)

    def note(*values: object,
             sep: str | None = " ",
             end: str | None = "\n",
             file: IO | None = None,
             flush: Literal[False] = False,
             tips: str = '注意',
             division=':',
             color=Fore.BLUE,
             ):
        if Log._show_note:
            color_print(*values, sep=sep, end=end, file=file,
                        flush=flush, tips=tips, color=color, division=division)

    def tips(*values: object,
             sep: str | None = " ",
             end: str | None = "\n",
             file: IO | None = None,
             flush: Literal[False] = False,
             tips: str = '提示',
             division=':',
             color=Fore.LIGHTBLACK_EX,
             ):
        if Log._show_tips:
            color_print(*values, sep=sep, end=end, file=file,
                        flush=flush, tips=tips, color=color, division=division)
