
from typing import List
from util.hangle_util import *
import re

class CustomLevin(object):

    def __init__(self) -> None:
        super().__init__()

        # COST 정보
        self.DEFAULT_SUB_COST = 0.9
        self.DEFAULT_INSERT_COST = 1.1
        self.DEFAULT_DEL_COST = 1.1

        self.COST = dict()
        self.COST['sub'] = dict()
        self.COST['ins'] = dict()
        self.COST['del'] = dict()

        for k in self.COST:
            self.COST[k] = dict()
        # sub cost 설정
        self.subcost = self.COST['sub']
        self.insertcost = self.COST['ins']
        self.deletecost = self.COST['del']

        # sub cost 설정
        self.subcost[('a', 's')] = 0.5
        self.subcost[('a', 'd')] = 0.5  # a를 d로 잘못 치는 경우를 숫자로 표현
        self.subcost[('a', 'f')] = 0.5
        self.subcost[('a', 'l')] = 2
        self.subcost[('a', 'q')] = 1.2
        self.subcost[('a', 'z')] = 1.2
        self.subcost[('ㅅ', 'ㅆ')] = 0.7

    # 두개의 STRING에 대한 거리값
    def get_string_distance(self, s1, s2, method='jamo', debug=False):
        # s1, s2에 대한 보정을 해야한다.
        ret = 0
        len_coef = len(s1) / len(s2)
        if method == 'jamo':
            ret = self.jamo_levenshtein(s1, s2, debug)
        else:
            ret = self.levenshtein(s1, s2, self.COST, debug)
        return ret * len(s1)

    # 하나의 string을 decompose
    def decompose_string(self, s: str):
        # 하나의 string을 decompose
        ret = []
        for i in s:
            ret.append(self.decompose(i))
        return ret

    # 하나의 string을 compose
    def compose_string(self, decomposed_li):
        # 분할된 encoded list(string 변환된가)
        ret = ''
        for i in decomposed_li:
            ret += self.compose_enc_char(i)
        return ret

    # character 단위 교체, 삭제, 삽입 비용
    def substitution_cost(self, c1, c2):
        if c1 == c2:
            return 0
        # 이거 비교할 떄, decompose(c1) , decompose(c2) 모두 string 형태로 바꿔어야 하는거아닌가?
        if self.character_is_korean(c1) and self.character_is_korean(c2):
            fc1 = self._flat(self.decompose(c1))
            fc2 = self._flat(self.decompose(c2))
            # print("qwer", c1, c2, self.levenshtein(fc1, fc2) / 9)
            return self.levenshtein(fc1, fc2) / 3
        else:
            # print("DFS", c1, c2, cost['sub'].get((c1, c2), DEFAULT_SUB_COST))

            return self.COST['sub'].get((c1, c2), self.DEFAULT_SUB_COST)

    def deletion_cost(self, c1, c2):
        return self.COST['del'].get((c1, c2), self.DEFAULT_DEL_COST)

    def insertion_cost(self, c1, c2, ):
        return self.COST['ins'].get((c1, c2), self.DEFAULT_INSERT_COST)

    def jamo_levenshtein(self, s1, s2, debug=False):
        if len(s1) < len(s2):
            return self.jamo_levenshtein(s2, s1, debug)
        if len(s2) == 0: return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + self.DEFAULT_INSERT_COST
                deletions = current_row[j] + self.DEFAULT_DEL_COST
                # Changed
                substitutions = previous_row[j] + self.substitution_cost(c1, c2)
                current_row.append(min(insertions, deletions, substitutions))
            if debug:
                print(['%.3f' % v for v in current_row[1:]])

            previous_row = current_row
        return previous_row[-1]

    def levenshtein(self, s1, s2, debug=False):
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1, debug=debug)
        if len(s2) == 0: return len(s1)
        cost = self.COST
        if cost is None:
            cost = dict()
            cost['sub'] = dict()
            cost['ins'] = dict()
            cost['del'] = dict()
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + cost['ins'].get((c1, c2), self.DEFAULT_SUB_COST)
                deletions = current_row[j] + cost['del'].get((c1, c2), self.DEFAULT_DEL_COST)
                substitutions = previous_row[j] + cost['sub'].get((c1, c2), self.DEFAULT_INSERT_COST)
                current_row.append(min(insertions, deletions, substitutions))
            if debug:
                print(current_row[1:])
            previous_row = current_row
        return previous_row[-1]

    def character_is_korean(self, c: str):
        # c는 문자 하나
        i = ord(c)
        return ((kor_begin <= i <= kor_end) or
                (jaum_begin <= i <= jaum_end) or
                (moum_begin <= i <= moum_end))

    def compose_enc_char(self, li: List[str]):
        # todo: li에 대한 예외처리
        if len(li) != 3:
            ret = ''
            for i in li:
                ret += str(i)
            return ret
        else:
            return self.compose(li[0], li[1], li[2])

    def compose(self, chosung, jungsung, jongsung):
        # 초성, 중성, 종성 각자 하나만 있는 경우
        # 초성 중성
        # 초성 중성 종성

        if chosung == ' ':
            if jungsung in jungsung_list:
                char = jungsung
            else:  # ' ', ' ', ' ',
                raise ValueError
        elif jungsung == ' ':
            char = chosung
        else:
            char = chr(
                kor_begin +
                chosung_base * chosung_list.index(chosung) +
                jungsung_base * jungsung_list.index(jungsung) +
                jongsung_list.index(jongsung)
            )
        return char

    def decompose(self, c):
        # 하나의 문자를 분해하기
        # 한글이면 3개 분할, 아니면 자기자신으로만

        if not self.character_is_korean(c):
            return (c,)
        i = ord(c)
        if (jaum_begin <= i <= jaum_end):
            return (c, ' ', ' ')
        if (moum_begin <= i <= moum_end):
            return (' ', c, ' ')

        # decomposition rule
        i -= kor_begin
        cho = i // chosung_base
        jung = (i - cho * chosung_base) // jungsung_base
        jong = (i - cho * chosung_base - jung * jungsung_base)
        return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])

    def _flat(self, target):
        # 분할된 문자, 튜플, 을 string으로 바꾸기
        ret = ''
        for i in target: ret += i
        if len(i) == 1: ret += '  '
        return ret

    # 분리 된 것에 대해 비교를 한다.

    def _flat_string(self, decompose_str):
        ret = ''
        for d in decompose_str:
            for i in d:
                if len(d) != 1:
                    if i != ' ':
                        ret += i
                elif len(d) == 1:
                    ret += i
        return ret

    def decompose_string_with_flat(self, target) -> str:
        return self._flat_string(self.decompose_string(target))




if __name__ == '__main__':
    corr = Corrector()
    print(corr.get_string_distance('나이키 태크', '나이키'))