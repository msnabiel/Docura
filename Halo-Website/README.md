# Swipe-AI Interview Platform

Swipe-AI is an AI-powered interview management platform that allows candidates to upload resumes, answer dynamically generated technical questions, and track their interview progress in real-time. The app ensures a smooth and persistent experience, saving timers, answers, and progress locally so candidates can continue where they left off.

## Flow
![Swipe-AI UML Diagram](/public/swipe-flow.png)
# Screenshots
![Swipe-AI UML Diagram](/public/homepage.png)
![Swipe-AI UML Diagram](/public/resume_upload.png)
![Swipe-AI UML Diagram](/public/questions.png)
![Swipe-AI UML Diagram](/public/welcomeback.png)    
![Swipe-AI UML Diagram](/public/dashboard2.png)

## **Features**

### **Candidate Experience**

* **Resume Upload**: Candidates upload their resume in PDF or DOCX format.
* **AI-Powered Parsing**: The system extracts name, email, and phone number automatically.
* **Missing Info**: Asks for missing info, if any.
* **Text Extration**: Python Backend for text extraction. (https://docura-parser.onrender.com) Github: https://github.com/msnabiel/docura-parser
* **Dynamic Interview Questions**: Interview questions are generated via AI (Gemini API) based on role and difficulty.
* **Timed Questions**: Each question has a timer depending on difficulty (Easy: 20s, Medium: 60s, Hard: 120s).
* **Answer Submission**: Candidates submit answers which are scored automatically by AI.
* **Progress Persistence**: All timers, answers, and question indices are saved locally.
* **Welcome Back Modal**: If a candidate refreshes or closes the page, they are welcomed back to continue the unfinished interview.

### **Admin / Recruiter**

* **Candidate Tracking**: Scores, answers, and summaries are stored in Redux for tracking.
* **Final Summary**: After the interview, a summarized score and chat history is saved for review.



## **Tech Stack**

* **Frontend**: React, Next.js (App Router), TypeScript
* **State Management**: Redux Toolkit + redux-persist
* **UI Library**: TailwindCSS + Shadcn/ui
* **AI Integration**: Gemini API for parsing resumes, generating questions, and scoring answers
* **Persistence**: LocalStorage (via redux-persist)



## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/msnabiel/Swipe-AI.git
cd Swipe-AI
```

2. Install dependencies:

```bash
npm install
```

3. Set environment variables:

```env
GOOGLE_KEY=<Your Gemini API endpoint>
```

4. Run the development server:

```bash
npm run dev
```



## **Usage**

1. Open the app in your browser at `http://localhost:3000`.
2. Upload your resume to start the interview.
3. Fill in any missing candidate info (name, email, phone).
4. Answer AI-generated questions within the timer limit.
5. If you refresh the page or return later, the app will restore your progress and show a “Welcome Back” modal.
6. Upon completion, your final score and summary are stored.



## **Future Improvements**

* Support multiple roles dynamically.
* Add authentication for candidates.
* Store interview sessions in a backend database for recruiters.
* Add AI-powered feedback for each answer.



## **Contributing**

Contributions are welcome!

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes and commit: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request



## **License**

MIT License © 2025 Nabiel


