import operator
from typing import Annotated, Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from logging import getLogger, basicConfig, DEBUG, INFO, CRITICAL
import os

import yaml
from pathlib import Path

"""
時系列調書作成※サンプル用意済み
↓
ペルソナを生成（陪審員や感情を扱う一般人）
なんならＭＢＴＩやビッグファイブに基づいてもいいかも
裁判官、論理学者、哲学者、推理小説家、恋愛小説家など職種でもいいかも
↓
それぞれのペルソナにインタビュー
↓
評価→甘いならペルソナ再作成
↓
レポートと解決策を提案した書類を生成

作成物
① DataModel（データクラス）
② State（途中経過を保持）
③ PersonaGenerator（ペルソナ生成）
④ InterviewGenerator（インタビュー）
⑤ Evaluator（評価）
⑥ ReportGenerator（レポート生成）
⑦ Conductor（全体の流れをコントロール）
⑧ Agent（LangGraph の実体）
"""
# class RecordDocument(BaseModel):
#   user_name = str = Field(..., description="ユーザの名前が入ります。AやBなどのときもあります。")
  
import yaml
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# ============================================================
# ① DataModel（ペルソナ / Big5 / Emotion / Interview）
# ============================================================

class BigFive(BaseModel):
    openness: float = Field(..., description="開放性")
    conscientiousness: float = Field(..., description="誠実性")
    extraversion: float = Field(..., description="外向性")
    agreeableness: float = Field(..., description="協調性")
    neuroticism: float = Field(..., description="神経症傾向")


class PlutchikEmotion(BaseModel):
    joy: float = Field(..., description="喜び")
    trust: float = Field(..., description="信頼")
    fear: float = Field(..., description="恐れ")
    surprise: float = Field(..., description="驚き")
    sadness: float = Field(..., description="悲しみ")
    disgust: float = Field(..., description="嫌悪")
    anger: float = Field(..., description="怒り")
    anticipation: float = Field(..., description="予期")


class HumanType(BaseModel):
    mbti: str = Field(..., description="MBTIタイプ")
    big5: BigFive = Field(..., description="Big Five（5因子）")
    occupation: str = Field(..., description="職業")
    emotion: PlutchikEmotion = Field(..., description="8感情プロファイル")


class Persona(BaseModel):
    name: str = Field(..., description="ペルソナ名")
    background: str = Field(..., description="背景設定")
    analysis: HumanType = Field(..., description="MBTI・Big5・感情など分析データ")


class Personas(BaseModel):
    personas: list[Persona] = Field(default_factory=list, description="ペルソナリスト")


class Interview(BaseModel):
    persona: Persona = Field(..., description="ペルソナ")
    question: str = Field(..., description="質問文")
    answer: str = Field(..., description="回答文")


class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(default_factory=list)


# ============================================================
# ② PersonaGenerator（デフォルト+LLM生成）
# ============================================================

class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5, yaml_path="config/personas.yml"):
        self.llm = llm.with_structured_output(Personas)
        self.k = k
        self.yaml_path = yaml_path

        # YAMLからデフォルトペルソナをロード
        self.default_personas = self._load_default_personas()

    def _load_default_personas(self) -> list[Persona]:
        path = Path(self.yaml_path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        personas = []
        for p in data.get("default_personas", []):
            personas.append(
                Persona(
                    name=p["name"],
                    background=p["background"],
                    analysis=HumanType(
                        mbti=p["analysis"]["mbti"],
                        occupation=p["analysis"]["occupation"],
                        big5=BigFive(**p["analysis"]["big5"]),
                        emotion=PlutchikEmotion(**p["analysis"]["emotion"])
                    )
                )
            )
        return personas

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "あなたは多様なペルソナ生成の専門家です。\n"
                        "すでに4つのデフォルトペルソナが存在しているため、"
                        f"あなたは追加で {self.k} 名のみ生成します。\n"
                        "構造化出力（Personasモデル）で返答してください。"
                    )
                ),
                (
                    "user",
                    "ユーザー依頼内容:\n----\n{user_request}\n----"
                )
            ]
        )

        chain = prompt | self.llm
        generated = chain.invoke({"user_request": user_request})

        # デフォルト + LLM生成を結合
        all_personas = self.default_personas + generated.personas

        return Personas(personas=all_personas)


# ============================================================
# ③ InterviewGenerator（質問→回答→InterviewResult）
# ============================================================

class InterviewGenerator:
    def __init__(self, llm: ChatOpenAI, questions_per_persona: int = 3):
        self.llm = llm
        self.questions_per_persona = questions_per_persona

    # メイン実行
    def run(self, user_request: str, personas: list[Persona], record_text: str) -> InterviewResult:
        questions = self._generate_questions(user_request, personas, record_text)
        answers = self._generate_answers(personas, questions, record_text)
        interviews = self._create_interviews(personas, questions, answers)
        return InterviewResult(interviews=interviews)

    # 質問生成
    def _generate_questions(self, user_request: str, personas: list[Persona], record_text: str) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはインタビュー心理学の専門家です。"
                    "調書を読み、各ペルソナが関心を持つ論点をもとに質問を生成してください。"
                ),
                (
                    "user",
                    (
                        "【ユーザー依頼内容】\n"
                        f"{user_request}\n\n"
                        "【調書内容】\n"
                        f"{record_text}\n\n"
                        f"各ペルソナにつき {self.questions_per_persona} 個の質問を生成してください。\n"
                        "【対象ペルソナ一覧】\n{personas}"
                    )
                )
            ]
        )

        chain = prompt | self.llm
        result = chain.invoke({
            "user_request": user_request,
            "record": record_text,
            "personas": [p.dict() for p in personas]
        })

        return result

    # 回答生成
    def _generate_answers(self, personas: list[Persona], questions: dict, record_text: str) -> dict:
        answers = {}

        for persona in personas:
            qs = questions.get(persona.name, [])

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "あなたは指定されたペルソナになりきり回答します。"
                        "調書内容を元に、価値観・感情・属性に忠実に答えてください。"
                    ),
                    (
                        "user",
                        (
                            f"【ペルソナ情報】\n{persona.dict()}\n\n"
                            f"【調書】\n{record_text}\n\n"
                            f"【質問】\n{qs}\n\n"
                            "順に回答してください。"
                        )
                    )
                ]
            )

            chain = prompt | self.llm
            response = chain.invoke({
                "persona": persona.dict(),
                "questions": qs,
                "record": record_text
            })

            answers[persona.name] = response

        return answers

    # Interview オブジェクト作成
    def _create_interviews(self, personas: list[Persona], questions: dict, answers: dict) -> list[Interview]:
        interview_list = []
        for persona in personas:
            qs = questions.get(persona.name, [])
            ans = answers.get(persona.name, [])
            for q, a in zip(qs, ans):
                interview_list.append(
                    Interview(persona=persona, question=q, answer=a)
                )
        return interview_list


# ============================================================
# ④ RecordLoader（任意：3視点の調書フォーマットをパース）
# ============================================================

class RecordLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> dict:
        """
        return:
        {
          "view_1": "...",
          "view_2": "...",
          "view_3": "...",
          "full_text": "...全部結合..."
        }
        """
        text = Path(self.filepath).read_text(encoding="utf-8")

        sections = {
            "view_1": "",
            "view_2": "",
            "view_3": "",
        }

        current = None

        for line in text.splitlines():
            line = line.strip()
            if "視点1" in line:
                current = "view_1"
                continue
            if "視点2" in line:
                current = "view_2"
                continue
            if "視点3" in line:
                current = "view_3"
                continue

            if current:
                sections[current] += line + "\n"

        return {
            **sections,
            "full_text": sections["view_1"] + sections["view_2"] + sections["view_3"]
        }




class DocumentationAgent:
  # def __init__(self):
  #   # 各じぇねらーたの初期化
    
  #   # グラフの作成
  #   self.graph = self._create_graph()
    
  #   def _create_graph(self) -> StateGraph:
  #     # グラフの初期化
  #     workflow = StateGraph(~State)
      
  #     #各ノードの追加
  #     # ペルソナ生成
  #     # インタビュー実施→判断実施に変わる
  #     # 情報の評価
  #     # 文章の作成
      
  #     #エントリーポイントの設定
  #     workflow.set_entry_point("generate_personas")
      
  #     # ノード間のエッジのついあｋ
  #     # 時系列調書は用意する
  #     workflow.add_edge("generate_personas", "判断実施の何かを入れる")
  #     workflow.add_edge("判断実施の何か", "まとめるための何か")
  #     workflow.add_edge("conduct_interviews", "evaluate_information")
      
      
  #     # 条件付きエッジの追加
      
  #     # グラフのコンパイル
  #     return workflow.compile()
    
  #   def _generate_personas(self, state:)

    pass

"""
設定やデバック関連の変数一覧
"""
logger = getLogger(__name__)
def debug_settings(debug_level=CRITICAL):
  basicConfig(level=debug_level, format="%(levelname)s: %(message)s")

# 環境変数の読み込みと各種APIキーの確認
def load_env():
    # GitHub Actions では dotenv を使わない
    if os.getenv("GITHUB_ACTIONS") == "true":
        logger.debug("GitHub Actions 認識 → .env は読み込まない")
    else:
        if load_dotenv():
            logger.debug(".env の読み込み：成功")
        else:
            logger.warning(".env の読み込み：失敗")

    # 共通で必要な環境変数
    required_keys = ["OPENAI_API_KEY"]

    missing = [k for k in required_keys if os.getenv(k) is None]
    if missing:
        logger.warning(f"環境変数が不足しています: {missing}")
        return False

    return True


# いったんユースケース駆動
if __name__ == '__main__':
    # デバック用のログレベルを設定
    debug_settings()

    # 環境変数の読み込み
    if not load_env():
        raise SystemExit("環境変数の読み込みに失敗しました。")

    # ====== LLM 初期化 ======
    llm = ChatOpenAI(model="gpt-4o-mini")

    # ====== 1. 調書の読み込み ======
    record_loader = RecordLoader("record-document-example.txt")
    record_data = record_loader.load()
    record_text = record_data["full_text"]

    # ====== 2. ペルソナ生成 ======
    pgen = PersonaGenerator(
        llm=llm,
        k=2,                      # 追加で何人生成するか
        yaml_path="config/personas.yml"
    )

    personas_obj = pgen.run(
        user_request="このトラブルの原因を多角的に分析するためのペルソナを生成してください。"
    )

    personas = personas_obj.personas  # list[Persona]

    print("\n===== 生成されたペルソナ一覧 =====")
    for p in personas:
        print(f"- {p.name} ({p.analysis.mbti}, {p.analysis.occupation})")


    # ====== 3. インタビュー生成 ======
    igen = InterviewGenerator(
        llm=llm,
        questions_per_persona=2   # 1人当たりの質問数
    )

    interview_result = igen.run(
        user_request="今回の映画デートのトラブルの原因を掘り下げたい。",
        personas=personas,
        record_text=record_text
    )

    # ====== 4. 結果出力 ======
    print("\n===== インタビュー結果 =====")
    for item in interview_result.interviews:
        print("\n------------------------------------")
        print(f"[ペルソナ] {item.persona.name}")
        print(f"[質問]\n{item.question}")
        print(f"[回答]\n{item.answer}")
