import streamlit as st
import json
import os
import random
import time
import tempfile
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI College MCQ Exam",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 25px;
}

.exam-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}

.question-title {
    font-size: 20px;
    font-weight: 700;
}

.mark-badge {
    padding: 5px 10px;
    border-radius: 10px;
    background-color: #eee;
    font-weight: 600;
}

.result-box {
    padding: 25px;
    border-radius: 20px;
    border: 2px solid #ddd;
    text-align: center;
}

.big-score {
    font-size: 48px;
    font-weight: 800;
}

.small-text {
    color: #666;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "questions": [],
    "answers": {},
    "exam_started": False,
    "exam_submitted": False,
    "exam_start_time": None,
    "exam_duration": 0,
    "uploaded_file_name": None,
    "uploaded_file_bytes": None,
    "pdf_file": None,
    "total_marks": 30,
    "exam_result": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Check your .env file."
        )

    return genai.Client(api_key=api_key)


# ============================================================
# GENERATE MCQs FROM PDF
# ============================================================

def generate_mcqs(pdf_file, total_marks, question_types):

    client = get_gemini_client()

    # --------------------------------------------------------
    # DECIDE NUMBER OF QUESTIONS
    # --------------------------------------------------------

    if total_marks == 30:

        if question_types == ["1 Mark"]:
            number_of_questions = 30

        elif question_types == ["2 Marks"]:
            number_of_questions = 15

        else:
            number_of_questions = 20

    else:

        if question_types == ["1 Mark"]:
            number_of_questions = 50

        elif question_types == ["2 Marks"]:
            number_of_questions = 25

        else:
            number_of_questions = 32


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an expert college examination paper setter.

The uploaded PDF is the ONLY study material.

Create an MCQ examination from the uploaded PDF.

EXAM DETAILS:

Total marks: {total_marks}

Allowed question types:
{question_types}

Number of questions to generate:
{number_of_questions}

IMPORTANT RULES:

1. Questions MUST be based only on information contained in the PDF.

2. Do not introduce unrelated outside knowledge.

3. Questions should be suitable for a college-level examination.

4. Questions should test:
   - definitions
   - concepts
   - important facts
   - differences
   - applications
   - examples
   - terminology
   - conceptual understanding

5. Avoid extremely easy questions.

6. Avoid ambiguous questions.

7. Avoid duplicate questions.

8. Every question must have exactly four options.

9. There must be exactly one correct answer.

10. Each question must have either 1 or 2 marks.

11. Give a short explanation for the correct answer.

12. Make the questions similar to questions that could appear in a
college internal/mid-semester examination.

13. Do NOT reveal the answer in the question itself.

14. Use ONLY the allowed question types selected by the user.

15. The final questions MUST have a total mark value of EXACTLY
{total_marks}.

VERY IMPORTANT:

Return ONLY valid JSON.

Do not use Markdown.

Do not use ```json.

JSON FORMAT:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": {{
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D"
            }},
            "correct_answer": "A",
            "marks": 1,
            "explanation": "Short explanation."
        }}
    ]
}}

The final questions must have a total mark value of EXACTLY {total_marks}.
"""


    # --------------------------------------------------------
    # CONVERT STREAMLIT PDF TO BYTES
    # --------------------------------------------------------

    pdf_bytes = pdf_file.getvalue()

    temp_pdf_path = None
    uploaded_pdf = None
    response = None


    try:

        # ----------------------------------------------------
        # SAVE PDF TEMPORARILY
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_pdf:

            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name


        # ----------------------------------------------------
        # UPLOAD PDF TO GEMINI
        # ----------------------------------------------------

        uploaded_pdf = client.files.upload(
            file=temp_pdf_path
        )


        # ----------------------------------------------------
        # GENERATE MCQs
        # RETRY IF GEMINI RETURNS 503
        # ----------------------------------------------------

        for attempt in range(4):

            try:

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=[
                        prompt,
                        uploaded_pdf
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )

                # Generation successful
                break


            except Exception as e:

                error_text = str(e)

                # --------------------------------------------
                # TEMPORARY SERVER ERROR
                # --------------------------------------------

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    if attempt < 3:

                        wait_time = 10 * (2 ** attempt)

                        st.warning(
                            f"Gemini is temporarily busy. "
                            f"Retrying in {wait_time} seconds..."
                        )

                        time.sleep(wait_time)

                    else:

                        raise


                # --------------------------------------------
                # OTHER ERROR
                # --------------------------------------------

                else:

                    raise


    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY PDF
        # ----------------------------------------------------

        if (
            temp_pdf_path
            and os.path.exists(temp_pdf_path)
        ):

            os.remove(temp_pdf_path)


    # --------------------------------------------------------
    # CHECK RESPONSE
    # --------------------------------------------------------

    if response is None:

        raise ValueError(
            "Gemini did not return a response."
        )


    if not response.text:

        raise ValueError(
            "Gemini returned an empty response."
        )


    # --------------------------------------------------------
    # CONVERT JSON RESPONSE
    # --------------------------------------------------------

    try:

        data = json.loads(response.text)

    except json.JSONDecodeError as e:

        raise ValueError(
            "Gemini returned invalid JSON."
        ) from e


    questions = data.get("questions", [])


    if not questions:

        raise ValueError(
            "No questions were generated."
        )


    # ========================================================
    # VALIDATE QUESTIONS
    # ========================================================

    valid_questions = []

    for q in questions:

        if not isinstance(q, dict):
            continue


        question_text = q.get("question")

        options = q.get("options")

        correct = q.get("correct_answer")

        marks = q.get("marks")

        explanation = q.get(
            "explanation",
            ""
        )


        # ----------------------------------------------------
        # QUESTION TEXT
        # ----------------------------------------------------

        if not question_text:
            continue


        # ----------------------------------------------------
        # OPTIONS
        # ----------------------------------------------------

        if not isinstance(options, dict):
            continue


        required_options = [
            "A",
            "B",
            "C",
            "D"
        ]


        if not all(
            option in options
            for option in required_options
        ):

            continue


        # ----------------------------------------------------
        # CORRECT ANSWER
        # ----------------------------------------------------

        if correct not in required_options:
            continue


        # ----------------------------------------------------
        # MARKS
        # ----------------------------------------------------

        if marks not in [1, 2]:
            continue


        # ----------------------------------------------------
        # RESPECT SELECTED QUESTION TYPES
        # ----------------------------------------------------

        if question_types == ["1 Mark"]:

            if marks != 1:
                continue


        elif question_types == ["2 Marks"]:

            if marks != 2:
                continue


        # ----------------------------------------------------
        # SAVE VALID QUESTION
        # ----------------------------------------------------

        valid_questions.append({

            "question": str(
                question_text
            ),

            "options": {

                "A": str(
                    options["A"]
                ),

                "B": str(
                    options["B"]
                ),

                "C": str(
                    options["C"]
                ),

                "D": str(
                    options["D"]
                )
            },

            "correct_answer": correct,

            "marks": int(marks),

            "explanation": str(
                explanation
            )
        })


    # ========================================================
    # CHECK NUMBER OF VALID QUESTIONS
    # ========================================================

    if not valid_questions:

        raise ValueError(
            "Gemini generated questions, "
            "but none passed validation."
        )


    # ========================================================
    # REMOVE DUPLICATE QUESTIONS
    # ========================================================

    unique_questions = []

    seen_questions = set()

    for q in valid_questions:

        question_key = (
            q["question"]
            .strip()
            .lower()
        )

        if question_key not in seen_questions:

            seen_questions.add(
                question_key
            )

            unique_questions.append(q)


    valid_questions = unique_questions


    # ========================================================
    # SHUFFLE QUESTIONS
    # ========================================================

    random.shuffle(
        valid_questions
    )


    # ========================================================
    # SELECT QUESTIONS
    # FOR EXACT TOTAL MARKS
    # ========================================================

    selected = []

    current_marks = 0


    # --------------------------------------------------------
    # FIRST SIMPLE APPROACH
    # --------------------------------------------------------

    for q in valid_questions:

        if (
            current_marks
            + q["marks"]
            <= total_marks
        ):

            selected.append(q)

            current_marks += q["marks"]


        if current_marks == total_marks:

            break


    # ========================================================
    # DYNAMIC PROGRAMMING IF NEEDED
    # ========================================================

    if current_marks != total_marks:

        dp = {
            0: []
        }


        for q in valid_questions:

            new_dp = dict(dp)


            for mark_sum, selected_questions in dp.items():

                new_sum = (
                    mark_sum
                    + q["marks"]
                )


                if (
                    new_sum <= total_marks
                    and new_sum not in new_dp
                ):

                    new_dp[new_sum] = (
                        selected_questions
                        + [q]
                    )


            dp = new_dp


            if total_marks in dp:

                break


        if total_marks in dp:

            selected = dp[
                total_marks
            ]


    # ========================================================
    # FINAL MARK CHECK
    # ========================================================

    selected_marks = sum(
        q["marks"]
        for q in selected
    )


    if selected_marks != total_marks:

        raise ValueError(
            "The AI did not generate enough "
            "valid questions to create an exact "
            f"{total_marks}-mark paper. "
            "Please click 'Generate Exam' again."
        )


    # ========================================================
    # SHUFFLE FINAL QUESTIONS
    # ========================================================

    random.shuffle(
        selected
    )


    return selected


# ============================================================
# CALCULATE RESULT
# ============================================================

def calculate_result():

    questions = st.session_state.questions

    answers = st.session_state.answers


    total_marks = 0

    obtained_marks = 0


    correct_count = 0

    wrong_count = 0

    unanswered_count = 0


    results = []


    # --------------------------------------------------------
    # CHECK EACH QUESTION
    # --------------------------------------------------------

    for index, q in enumerate(questions):

        marks = q["marks"]

        total_marks += marks


        user_answer = answers.get(
            index
        )

        correct_answer = (
            q["correct_answer"]
        )


        # ----------------------------------------------------
        # UNANSWERED
        # ----------------------------------------------------

        if user_answer is None:

            status = "Unanswered"

            unanswered_count += 1


        # ----------------------------------------------------
        # CORRECT
        # ----------------------------------------------------

        elif user_answer == correct_answer:

            status = "Correct"

            correct_count += 1

            obtained_marks += marks


        # ----------------------------------------------------
        # WRONG
        # ----------------------------------------------------

        else:

            status = "Wrong"

            wrong_count += 1


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({

            "question_number": index + 1,

            "question": q["question"],

            "options": q["options"],

            "user_answer": user_answer,

            "correct_answer": correct_answer,

            "marks": marks,

            "status": status,

            "explanation": q["explanation"]
        })


    # ========================================================
    # PERCENTAGE
    # ========================================================

    percentage = (

        obtained_marks
        / total_marks
        * 100

        if total_marks > 0

        else 0
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    answered_questions = (
        correct_count
        + wrong_count
    )


    accuracy = (

        correct_count
        / answered_questions
        * 100

        if answered_questions > 0

        else 0
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "total_marks": total_marks,

        "obtained_marks": obtained_marks,

        "percentage": percentage,

        "accuracy": accuracy,

        "correct": correct_count,

        "wrong": wrong_count,

        "unanswered": unanswered_count,

        "results": results
    }


# ============================================================
# SUBMIT EXAM
# ============================================================

def submit_exam():

    result = calculate_result()

    st.session_state.exam_result = result

    st.session_state.exam_submitted = True

    st.session_state.exam_started = False


# ============================================================
# START NEW EXAM
# ============================================================

def start_exam():

    st.session_state.answers = {}

    st.session_state.exam_submitted = False

    st.session_state.exam_started = True

    st.session_state.exam_start_time = (
        time.time()
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎓 AI College MCQ Exam'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    'Upload your PDF → Generate MCQs → '
    'Give Exam → Check Your Result'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Exam Settings"
    )


    # --------------------------------------------------------
    # TOTAL MARKS
    # --------------------------------------------------------

    total_marks = st.radio(

        "Select Exam Marks",

        [30, 50],

        index=0
    )


    # --------------------------------------------------------
    # QUESTION TYPES
    # --------------------------------------------------------

    question_types = st.multiselect(

        "Question Types",

        [
            "1 Mark",
            "2 Marks"
        ],

        default=[
            "1 Mark",
            "2 Marks"
        ]
    )


    st.divider()


    st.info(
        "The questions are generated "
        "from the uploaded PDF. "
        "The AI is instructed to use "
        "the PDF as the study source."
    )


    st.divider()


    st.caption(
        "🎓 AI College MCQ Exam"
    )

    st.caption(
        "Python + Streamlit + Gemini"
    )


# ============================================================
# RESULT PAGE
# ============================================================

if st.session_state.exam_submitted:

    result = (
        st.session_state.exam_result
    )


    st.success(
        "🎉 Exam Submitted Successfully!"
    )


    st.markdown(
        "## 📊 Your Result"
    )


    # --------------------------------------------------------
    # MAIN RESULT METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(

            "Marks",

            f"{result['obtained_marks']} "
            f"/ {result['total_marks']}"
        )


    with col2:

        st.metric(

            "Percentage",

            f"{result['percentage']:.1f}%"
        )


    with col3:

        st.metric(

            "Accuracy",

            f"{result['accuracy']:.1f}%"
        )


    with col4:

        st.metric(

            "Correct",

            result["correct"]
        )


    st.divider()


    # --------------------------------------------------------
    # QUESTION COUNTS
    # --------------------------------------------------------

    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        st.metric(
            "✅ Correct",
            result["correct"]
        )


    with col2:

        st.metric(
            "❌ Wrong",
            result["wrong"]
        )


    with col3:

        st.metric(
            "⚪ Unanswered",
            result["unanswered"]
        )


    st.divider()


    # --------------------------------------------------------
    # QUESTION REVIEW
    # --------------------------------------------------------

    st.header(
        "📚 Question Review"
    )


    for item in result["results"]:

        qno = item[
            "question_number"
        ]


        # ----------------------------------------------------
        # STATUS ICON
        # ----------------------------------------------------

        if item["status"] == "Correct":

            icon = "✅"

        elif item["status"] == "Wrong":

            icon = "❌"

        else:

            icon = "⚪"


        st.markdown(
            f"### {icon} Question {qno} "
            f"({item['marks']} Mark)"
        )


        st.write(
            item["question"]
        )


        # ----------------------------------------------------
        # SHOW OPTIONS
        # ----------------------------------------------------

        for letter in [
            "A",
            "B",
            "C",
            "D"
        ]:

            option_text = (
                item["options"][letter]
            )


            if (
                letter
                == item["correct_answer"]
            ):

                st.success(
                    f"✅ {letter}. "
                    f"{option_text} "
                    f"— Correct Answer"
                )


            elif (
                letter
                == item["user_answer"]
            ):

                st.error(
                    f"❌ {letter}. "
                    f"{option_text} "
                    f"— Your Answer"
                )


            else:

                st.write(
                    f"{letter}. "
                    f"{option_text}"
                )


        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        if item["status"] != "Correct":

            st.info(
                "💡 Explanation: "
                f"{item['explanation']}"
            )


        st.divider()


    # --------------------------------------------------------
    # GENERATE NEW EXAM
    # --------------------------------------------------------

    if st.button(
        "🔄 Generate New Exam",
        use_container_width=True
    ):

        st.session_state.questions = []

        st.session_state.answers = {}

        st.session_state.exam_started = False

        st.session_state.exam_submitted = False

        st.session_state.exam_result = None

        st.session_state.exam_start_time = None

        st.rerun()


    st.stop()


# ============================================================
# EXAM PAGE
# ============================================================

if st.session_state.exam_started:

    questions = (
        st.session_state.questions
    )


    # ========================================================
    # TIMER
    # ========================================================

    total_exam_seconds = (
        st.session_state.exam_duration
    )


    elapsed = int(

        time.time()
        - st.session_state.exam_start_time
    )


    remaining = max(

        0,

        total_exam_seconds
        - elapsed
    )


    minutes = (
        remaining // 60
    )

    seconds = (
        remaining % 60
    )


    # --------------------------------------------------------
    # TIME OVER
    # --------------------------------------------------------

    if remaining <= 0:

        st.warning(
            "⏰ Time is over! "
            "Your exam has been submitted."
        )

        submit_exam()

        st.rerun()


    # --------------------------------------------------------
    # TIMER COLOR
    # --------------------------------------------------------

    timer_color = "green"


    if remaining <= 60:

        timer_color = "red"


    # --------------------------------------------------------
    # TIMER DISPLAY
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:15px;
            border-radius:15px;
            border:2px solid {timer_color};
            margin-bottom:20px;
        ">

            <div style="
                font-size:18px;
                font-weight:600;
            ">
                ⏱️ Time Remaining
            </div>

            <div style="
                font-size:36px;
                font-weight:800;
                color:{timer_color};
            ">
                {minutes:02d}:{seconds:02d}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # JAVASCRIPT REFRESH
    # ========================================================

    st.markdown(
        """
        <script>

        setTimeout(function(){

            window.location.reload();

        }, 1000);

        </script>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # EXAM TITLE
    # ========================================================

    st.markdown(
        "## 📝 College Examination"
    )


    st.write(
        f"**Total Marks:** "
        f"{sum(q['marks'] for q in questions)}"
    )


    st.write(
        f"**Total Questions:** "
        f"{len(questions)}"
    )


    st.divider()


    # ========================================================
    # QUESTIONS
    # ========================================================

    for index, q in enumerate(
        questions
    ):

        st.markdown(
            f"### Q{index + 1}. "
            f"{q['question']}"
        )


        st.caption(
            f"📌 {q['marks']} Mark"
        )


        # ----------------------------------------------------
        # OPTIONS
        # ----------------------------------------------------

        option_items = [

            (
                "A",
                q["options"]["A"]
            ),

            (
                "B",
                q["options"]["B"]
            ),

            (
                "C",
                q["options"]["C"]
            ),

            (
                "D",
                q["options"]["D"]
            )
        ]


        current_answer = (
            st.session_state.answers.get(
                index
            )
        )


        option_labels = [

            f"{letter}. {text}"

            for letter, text
            in option_items
        ]


        # ----------------------------------------------------
        # RADIO BUTTON
        # ----------------------------------------------------

        selected = st.radio(

            "Choose your answer:",

            option_labels,

            index=(

                option_labels.index(

                    f"{current_answer}. "
                    f"{q['options'][current_answer]}"

                )

                if (
                    current_answer
                    in q["options"]
                )

                else None
            ),

            key=f"question_{index}",

            label_visibility="collapsed"
        )


        # ----------------------------------------------------
        # SAVE ANSWER
        # ----------------------------------------------------

        if selected:

            selected_letter = (
                selected[0]
            )

            st.session_state.answers[
                index
            ] = selected_letter


        st.divider()


    # ========================================================
    # SUBMIT EXAM
    # ========================================================

    if st.button(

        "🚀 SUBMIT EXAM",

        type="primary",

        use_container_width=True
    ):

        submit_exam()

        st.rerun()


    st.stop()


# ============================================================
# HOME / PDF UPLOAD PAGE
# ============================================================

st.header(
    "📄 Step 1 — Upload Your Study PDF"
)


uploaded_file = st.file_uploader(

    "Choose your PDF",

    type=["pdf"],

    help="Upload your chapter/unit PDF."
)


# ============================================================
# PDF UPLOADED
# ============================================================

if uploaded_file:

    st.success(
        f"📄 Uploaded: "
        f"{uploaded_file.name}"
    )


    # --------------------------------------------------------
    # FILE SIZE
    # --------------------------------------------------------

    file_size_mb = (

        len(
            uploaded_file.getvalue()
        )

        / (1024 * 1024)
    )


    st.write(
        f"File size: "
        f"**{file_size_mb:.2f} MB**"
    )


    # --------------------------------------------------------
    # FILE SIZE CHECK
    # --------------------------------------------------------

    if file_size_mb > 50:

        st.error(
            "This PDF is larger than "
            "50 MB. Please upload a "
            "smaller PDF."
        )


    else:

        # ====================================================
        # STEP 2
        # ====================================================

        st.header(
            "📝 Step 2 — Choose Exam"
        )


        col1, col2 = (
            st.columns(2)
        )


        with col1:

            st.info(
                f"🎯 Exam: "
                f"**{total_marks} Marks**"
            )


        with col2:

            st.info(
                f"📚 Types: "
                f"**{', '.join(question_types)}**"
            )


        # ----------------------------------------------------
        # QUESTION TYPE CHECK
        # ----------------------------------------------------

        if not question_types:

            st.warning(
                "Please select at least "
                "one question type from "
                "the sidebar."
            )


        else:

            # =================================================
            # STEP 3
            # =================================================

            st.header(
                "🚀 Step 3 — Generate Exam"
            )


            if st.button(

                "🤖 GENERATE MCQ EXAM",

                type="primary",

                use_container_width=True
            ):

                with st.spinner(

                    "🤖 AI is reading your "
                    "PDF and creating the "
                    "examination..."
                ):

                    try:

                        # ------------------------------------
                        # GENERATE QUESTIONS
                        # ------------------------------------

                        questions = generate_mcqs(

                            uploaded_file,

                            total_marks,

                            question_types
                        )


                        # ------------------------------------
                        # SAVE QUESTIONS
                        # ------------------------------------

                        st.session_state.questions = (
                            questions
                        )


                        st.session_state.total_marks = (
                            total_marks
                        )


                        st.session_state.uploaded_file_name = (
                            uploaded_file.name
                        )


                        st.session_state.uploaded_file_bytes = (
                            uploaded_file.getvalue()
                        )


                        # ------------------------------------
                        # SUCCESS
                        # ------------------------------------

                        st.success(
                            f"✅ {len(questions)} "
                            f"questions generated!"
                        )


                        # ------------------------------------
                        # PREVIEW
                        # ------------------------------------

                        st.write(
                            "### 📋 Exam Preview"
                        )


                        total_generated_marks = sum(

                            q["marks"]

                            for q in questions
                        )


                        st.write(
                            f"Total marks: "
                            f"**{total_generated_marks}**"
                        )


                        # ------------------------------------
                        # EXAM TIME
                        # ------------------------------------
                        #
                        # 1.5 minutes per mark
                        #
                        # 30 marks = 45 minutes
                        # 50 marks = 75 minutes
                        #

                        exam_minutes = (
                            total_marks * 1.5
                        )


                        st.session_state.exam_duration = int(

                            exam_minutes
                            * 60
                        )


                    except Exception as e:

                        st.error(
                            "❌ Could not generate "
                            "the exam."
                        )


                        st.exception(e)


            # =================================================
            # START EXAM
            # =================================================

            if st.session_state.questions:

                st.divider()


                st.header(
                    "🎓 Your Exam Is Ready!"
                )


                total_marks_ready = sum(

                    q["marks"]

                    for q
                    in st.session_state.questions
                )


                st.success(

                    f"Exam: "
                    f"{total_marks_ready} Marks | "
                    f"{len(st.session_state.questions)} "
                    f"Questions"
                )


                exam_minutes = (

                    st.session_state.exam_duration
                    // 60
                )


                st.info(
                    f"⏱️ Exam Time: "
                    f"{exam_minutes} minutes"
                )


                if st.button(

                    "▶️ START EXAM",

                    type="primary",

                    use_container_width=True
                ):

                    start_exam()

                    st.rerun()


# ============================================================
# API KEY HELP
# ============================================================

st.divider()


with st.expander(
    "🔑 Gemini API Key Setup"
):

    st.write(
        "The app requires a Gemini API key "
        "to generate questions from the PDF."
    )


    st.write(
        "You can provide the key using "
        "an environment variable:"
    )


    st.code(
        "GEMINI_API_KEY=your_api_key_here"
    )


    st.write(
        "Then restart Streamlit."
    )