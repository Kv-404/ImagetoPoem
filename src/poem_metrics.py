"""Simple, explainable poem quality scores (used to pick training data + evaluate).

Each score is between 0 and 1:
  lines   - is it exactly 8 lines?
  rhyme   - share of line pairs that rhyme (best of AABB / ABAB per verse)
  keyword - share of caption words the poem talks about (does it fit the image?)
  mood    - does it use words that fit the mood?
"""
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # pkg_resources notice
import pronouncing  # CMU pronouncing dictionary

STOP = set("""a an the of on in at to with and or is are was were be by for from
its it this that there their his her some two one three into over under near
while as up down out next very""".split())

MOOD_WORDS = {
    "happy": "joy joyful smile smiles laugh laughter bright glad cheer sunny warm play delight golden sing gleam".split(),
    "sad": "tear tears alone lonely grey gray rain cold fade fades lost sorrow sigh empty silent ache dark weep".split(),
    "calm": "calm still quiet soft gentle peace peaceful slow hush rest breeze serene whisper drift still mist".split(),
    "energetic": "run race rush leap jump spin fast wild fire burst roar dance loud beat speed thunder alive".split(),
}


def poem_lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def format_8(text):
    """First 8 lines as 2 verses of 4 (used for training targets)."""
    lines = poem_lines(text)
    if len(lines) < 8:
        return "\n".join(lines)
    return "\n".join(lines[:4]) + "\n\n" + "\n".join(lines[4:8])


def last_word(line):
    words = re.findall(r"[a-zA-Z']+", line.lower())
    return words[-1].strip("'") if words else ""


def rhymes(a, b):
    if not a or not b or a == b:
        return False
    pa, pb = pronouncing.phones_for_word(a), pronouncing.phones_for_word(b)
    if pa and pb:  # same sound from the last stressed vowel onwards
        return pronouncing.rhyming_part(pa[0]) == pronouncing.rhyming_part(pb[0])
    return a[-3:] == b[-3:]  # unknown word: compare spelling


def rhyme_score(text):
    ends = [last_word(l) for l in poem_lines(text)][:8]
    if len(ends) < 4:
        return 0.0
    got, total = 0, 0
    for v in range(0, len(ends) - 3, 4):           # each 4-line verse
        e = ends[v:v + 4]
        aabb = rhymes(e[0], e[1]) + rhymes(e[2], e[3])
        abab = rhymes(e[0], e[2]) + rhymes(e[1], e[3])
        got += max(aabb, abab)
        total += 2
    return got / total


def keyword_score(text, caption):
    words = [w for w in re.findall(r"[a-z]+", caption.lower()) if w not in STOP and len(w) > 2]
    if not words:
        return 0.0
    poem = text.lower()
    return sum(w[:4] in poem for w in words) / len(words)   # 4-letter stem match


def mood_score(text, mood):
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return min(1.0, len(tokens & set(MOOD_WORDS[mood])) / 2)


def line_score(text):
    return 1.0 if len(poem_lines(text)) == 8 else 0.0


def score_poem(text, caption, mood):
    """All metrics + one weighted total (used to rank candidate poems)."""
    lines = poem_lines(text)
    s = {"lines": line_score(text), "n_lines": len(lines), "rhyme": rhyme_score(text),
         "keyword": keyword_score(text, caption), "mood": mood_score(text, mood)}
    penalty = 0.0
    if len(set(lines)) < len(lines):
        penalty += 0.3                                   # repeated line
    if any(len(l.split()) > 12 for l in lines):
        penalty += 0.2                                   # line too long
    if re.search(r"poem|here's|title|stanza|verse", text.lower()):
        penalty += 0.3                                   # talks about itself
    s["total"] = 0.4 * s["rhyme"] + 0.3 * s["keyword"] + 0.3 * s["mood"] - penalty
    return s
