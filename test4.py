import streamlit as st
import json
import re
import os
from groq import Groq
from gtts import gTTS
import tempfile
from tenacity import retry, stop_after_attempt, wait_fixed

# Initialize session state
if 'quiz' not in st.session_state:
    st.session_state.quiz = {
        'api_key': None,
        'user_details': {},
        'questions': [],
        'current_q': 0,
        'score': 0,
        'difficulty': 'beginner',
        'history': [],
        'feedback': '',
        'chat_history': [],
        'raw_response': None,
        'parsing_errors': [],
        'attempt_count': 0
    }

# System instructions with strict formatting
SYSTEM_INSTRUCTION = """
You are an expert quiz generator. Follow these RULES:
1. Generate MCQs in this EXACT format:
**QuestionX**
{
    'Question': '...',
    'Options': {
        'OptionA': '...',
        'OptionB': '...', 
        'OptionC': '...',
        'OptionD': '...'
    },
    'Answer': '...'
}
2. Use SINGLE quotes for keys/values
3. No extra text before/after questions
4. Answers must EXACTLY match one option value
5. Ensure UNIQUE, age-appropriate questions
6. Maintain consistent option casing
7. Avoid special characters
8. Ensure proper JSON formatting
9. Ensure proper comma separation between key-value pairs
10. Avoid trailing commas in JSON objects
11. Use proper JSON boolean values (true/false)
12. Maintain consistent quotation usage
13. Validate JSON syntax before responding

EXAMPLE:
**Question1**
{
    'Question': 'What is 2+2?',
    'Options': {
        'OptionA': '3',
        'OptionB': '4',
        'OptionC': '5',
        'OptionD': '6'
    },
    'Answer': '4'
}
"""

def get_groq_client():
    """Initialize Groq client with validation"""
    if not st.session_state.quiz['api_key']:
        st.warning("🔑 Please enter your Groq API key")
        st.stop()
    try:
        return Groq(api_key=st.session_state.quiz['api_key'])
    except Exception as e:
        st.error(f"❌ API Connection Error: {str(e)}")
        st.stop()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def generate_questions(prompt):
    """Generate questions with retry logic"""
    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=4000
        )
        raw_response = response.choices[0].message.content
        st.session_state.quiz['raw_response'] = raw_response
        return extract_questions(raw_response)
    except Exception as e:
        st.session_state.quiz['attempt_count'] += 1
        st.error(f"⚠️ Attempt {st.session_state.quiz['attempt_count']} failed: {str(e)}")
        return None

def extract_questions(response):
    """Extract and parse questions from API response with enhanced regex"""
    question_blocks = re.findall(r'\*\*Question\d+\*\*\s*({.*?})\s*(?=\*\*Question\d+\*\*|$)', response, re.DOTALL)
    
    questions = []
    for block in question_blocks:
        try:
            # Convert to valid JSON
            json_str = block.replace("'", '"')
            json_str = re.sub(r'(\w+)(\s*:\s*)', r'"\1"\2', json_str)  # Add quotes to keys
            question_data = json.loads(json_str)
            questions.append(question_data)
        except json.JSONDecodeError as e:
            st.session_state.quiz['parsing_errors'].append({
                'error_type': 'JSON Decode',
                'message': str(e),
                'block': block
            })
    return questions

def text_to_speech(text):
    """Convert text to audio with error handling"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts = gTTS(text=text, lang='en')
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"🔇 TTS Error: {str(e)}")
        return None

def user_details_form():
    """Collect user information"""
    with st.form("user_details"):
        st.session_state.quiz['user_details'] = {
            'name': st.text_input("Student Name"),
            'grade': st.number_input("Grade Level", min_value=1, max_value=12, value=5),
            'subject': st.selectbox("Subject", ["Math", "Science", "History", "English"]),
            'topic': st.text_input("Topic", placeholder="E.g., Fractions, Solar System"),
            'difficulty': st.select_slider("Difficulty", ['Beginner', 'Intermediate', 'Advanced']),
            'num_questions': st.slider("Number of Questions", 5, 20, 10)
        }
        if st.form_submit_button("🚀 Start Quiz"):
            generate_quiz()

def generate_quiz():
    """Generate new quiz questions"""
    details = st.session_state.quiz['user_details']
    prompt = f"""
    Generate {details['num_questions']} {details['subject']} questions about {details['topic']}
    for a {details['grade']}th grade student at {details['difficulty']} level.
    """
    
    with st.spinner("🧠 Generating questions..."):
        questions = generate_questions(prompt)
    
    if questions:
        st.session_state.quiz.update({
            'questions': questions,
            'current_q': 0,
            'score': 0,
            'history': [],
            'feedback': '',
            'attempt_count': 0
        })
        st.rerun()
    else:
        st.error("❌ Failed to generate questions. Please try a different topic or reduce question count.")

def show_question():
    """Display current question with audio"""
    q_idx = st.session_state.quiz['current_q']
    q = st.session_state.quiz['questions'][q_idx]
    
    st.subheader(f"❓ Question {q_idx + 1}")
    st.markdown(f"**{q['Question']}**")
    
    # Audio version
    if audio_file := text_to_speech(q['Question']):
        st.audio(audio_file)
        os.unlink(audio_file)
    
    # Answer selection using option values
    options = list(q['Options'].values())
    user_answer = st.radio("Options:", options, index=None, key=f"q{q_idx}")
    
    if st.button("✅ Submit Answer"):
        process_answer(q, user_answer)

def process_answer(q, user_answer):
    """Handle answer submission and progression"""
    is_correct = user_answer.strip() == q['Answer'].strip()
    
    st.session_state.quiz['history'].append({
        'question': q['Question'],
        'user_answer': user_answer,
        'correct_answer': q['Answer'],
        'is_correct': is_correct
    })
    
    if is_correct:
        st.session_state.quiz['score'] += 1
    
    if st.session_state.quiz['current_q'] < len(st.session_state.quiz['questions']) - 1:
        st.session_state.quiz['current_q'] += 1
        st.rerun()
    else:
        generate_feedback()

def generate_feedback():
    """Generate AI performance analysis"""
    details = st.session_state.quiz['user_details']
    prompt = f"""
    Analyze performance for {details['name']} (Grade {details['grade']}):
    - Subject: {details['subject']}
    - Topic: {details['topic']}
    - Score: {st.session_state.quiz['score']}/{len(st.session_state.quiz['questions'])}
    - Question History: {st.session_state.quiz['history']}
    
    Provide 200-word analysis covering:
    1. Strengths and weaknesses
    2. Key areas needing improvement
    3. Study recommendations
    4. Encouraging feedback
    """
    
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        st.session_state.quiz['feedback'] = response.choices[0].message.content
    except Exception as e:
        st.error(f"📝 Feedback Error: {str(e)}")

def debug_panel():
    """Show debugging information"""
    with st.expander("🐞 Debug Panel"):
        st.subheader("Raw API Response")
        st.code(st.session_state.quiz.get('raw_response', 'No response captured'))
        
        st.subheader("Parsing Errors")
        if st.session_state.quiz['parsing_errors']:
            for error in st.session_state.quiz['parsing_errors']:
                st.error(f"""
                **Question Error**  
                Type: {error['error_type']}  
                Message: {error['message']}
                """)
                st.code(error['block'])
        else:
            st.success("✅ No parsing errors detected")
        
        st.subheader("Session State")
        st.json(st.session_state.quiz)

def chat_interface():
    """Interactive study assistant"""
    st.subheader("💬 Study Assistant")
    
    for msg in st.session_state.quiz['chat_history']:
        st.chat_message("user" if msg['is_user'] else "assistant").write(msg['content'])
    
    if prompt := st.chat_input("Ask about the topic..."):
        st.session_state.quiz['chat_history'].append({'is_user': True, 'content': prompt})
        
        try:
            client = get_groq_client()
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}]
            )
            reply = response.choices[0].message.content
            st.session_state.quiz['chat_history'].append({'is_user': False, 'content': reply})
            st.rerun()
        except Exception as e:
            st.error(f"💬 Chat Error: {str(e)}")

# Main application flow
st.title("🎓 Smart Study Pro")
st.caption("Powered by Groq AI • Adaptive Learning System")

# API Key Input
st.session_state.quiz['api_key'] = st.text_input(
    "Enter Groq API Key:",
    type="password",
    help="Get from https://console.groq.com/keys"
)

if st.session_state.quiz['api_key']:
    if not st.session_state.quiz.get('user_details'):
        user_details_form()
    else:
        if st.session_state.quiz['questions']:
            show_question()
        else:
            user_details_form()

    if st.session_state.quiz.get('feedback'):
        st.subheader("📊 Performance Report")
        st.write(st.session_state.quiz['feedback'])
        
        st.subheader("📝 Question Review")
        for i, result in enumerate(st.session_state.quiz['history']):
            with st.expander(f"Question {i+1}: {result['question']}", expanded=False):
                st.markdown(f"""
                **Your Answer:** {result['user_answer'] or 'No answer'}  
                **Correct Answer:** {result['correct_answer']}  
                **Result:** {"✅ Correct" if result['is_correct'] else "❌ Incorrect"}
                """)
        
        if st.button("🔄 Retake Quiz"):
            st.session_state.quiz.update({
                'questions': [],
                'current_q': 0,
                'score': 0,
                'history': [],
                'feedback': ''
            })
            st.rerun()

    chat_interface()
    debug_panel()

# Footer
st.markdown("---")
st.markdown("**Tips:** • Start with simple topics • Check debug panel if issues occur • Refresh to start over")