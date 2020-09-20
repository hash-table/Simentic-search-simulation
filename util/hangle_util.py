import re
from typing import List



# 한글 캐릭터 정보
kor_begin = 44032
kor_end = 55203
chosung_base = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643
chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
                'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
                 'ㅡ', 'ㅢ', 'ㅣ']
jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
    'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
    'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
    'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ',
             'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
             'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
             'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']




hangle = 0
jaso = 1
number = 2
english = 3
unk = 4
type_list = (hangle,jaso,number,english,unk)



def check_token_type(target: str, choosed_type: int) -> bool:
    if choosed_type not in type_list:
        raise ValueError

    for i in target:
        if get_char_type(i) != choosed_type:
            return False
    return True


def check_number(target: str) -> bool:
    return check_token_type(target, number)


def check_engilsh(target: str) -> bool:
    return check_token_type(target, english)


def check_hangle(target: str) -> bool:
    return check_token_type(target, hangle)


def check_jaso(target: str) -> bool:
    return check_token_type(target, jaso)


def check_symbol(target: str) -> bool:
    return check_token_type(target, unk)


def get_char_type(one_char: str) -> int:
    uni_val = ord(one_char)
    if uni_val >= ord('가') and uni_val <= ord('힣'):
        return hangle
    elif (uni_val >= ord('ㄱ') and uni_val <= ord('ㅎ')) and (uni_val >= ord('ㅏ') and uni_val <= ord('ㅣ')):
        return jaso
    elif (uni_val >= ord('0') and uni_val <= ord('9')):
        return number
    elif (uni_val >= ord('a') and uni_val <= ord('z')) or (uni_val >= ord('A') and uni_val <= ord('Z')):
        return english
    else:
        return unk


def action(target: str, except_unk: bool = True) -> List[str]:
    no_char = r'[^가-힣ㄱ-ㅎ0-9a-zA-Z]'
    replace_char = ''
    # 1. target
    target = re.sub(no_char, '', target)
    # 2. classification
    ret = []
    UNK = 4
    prev_char_type = get_char_type(target[0])
    buf = [target[0]]

    for i in target[1:]:
        if prev_char_type == get_char_type(i):
            buf.append(i)
        else:
            if except_unk and prev_char_type != UNK:
                ret.append(''.join(buf))
            prev_char_type = get_char_type(i)
            buf = [i]
    if buf:
        if except_unk and prev_char_type != UNK:
            ret.append(''.join(buf))
    return ret



