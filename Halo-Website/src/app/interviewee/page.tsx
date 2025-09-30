'use client'

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { useDispatch, useSelector } from 'react-redux';
import { addCandidate } from '@/store/slices/candidateSlice';
import {
  setCandidateInfo,
  setQuestions,
  setAnswer,
  setCurrentQuestionIndex,
  setTimer,
  setInProgress,
  resetInterview,
  markCompleted,
  setDeadline
} from '@/store/slices/interviewSlice';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

interface ScoreResult {
  question: string;
  answer: string;
  score: number;
}

interface ScoreData {
  results: ScoreResult[];
  summary: string;
}

interface Question {
  text: string;
  difficulty: 'easy' | 'medium' | 'hard';
}

interface CandidateInfo {
  name?: string | null;
  email?: string | null;
  phone?: string | null;
}

interface RootState {
  interview: {
    candidateInfo: CandidateInfo;
    currentQuestionIndex: number;
    answers: string[];
    deadline: number | null;  
    timer: number;
    questions: Question[];
    inProgress: boolean;
  };
}

const IntervieweeTab: React.FC = () => {
  const dispatch = useDispatch();
  
  // Redux state
  const candidateInfo = useSelector((state: RootState) => state.interview.candidateInfo);
  const questions = useSelector((state: RootState) => state.interview.questions);
  const currentQuestionIndex = useSelector((state: RootState) => state.interview.currentQuestionIndex);
  const answers = useSelector((state: RootState) => state.interview.answers);
  const timer = useSelector((state: RootState) => state.interview.timer);
  const inProgress = useSelector((state: RootState) => state.interview.inProgress);
  const deadline = useSelector((state: RootState) => state.interview.deadline);
  // Local state
  const [file, setFile] = useState<File | null>(null);
  const [missingFields, setMissingFields] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [resumeParsed, setResumeParsed] = useState(false);
  const [answerInput, setAnswerInput] = useState('');
  const [showWelcomeBack, setShowWelcomeBack] = useState(false);
  const [hasCompletedScoring, setHasCompletedScoring] = useState(false);

  // Check for existing session on mount
  useEffect(() => {
    if (inProgress && questions.length > 0 && currentQuestionIndex < questions.length) {
      setShowWelcomeBack(true);
      setAnswerInput(answers[currentQuestionIndex] || '');
    }
  }, []);

  // Timer logic
// Timer logic with deadline
useEffect(() => {
  if (!questions.length || currentQuestionIndex >= questions.length || showWelcomeBack) return;

  const currentQuestion = questions[currentQuestionIndex];
  const timeForQuestion = getTimeByDifficulty(currentQuestion.difficulty);

  // If no deadline set, create one
  if (!timer || !inProgress) {
    const newDeadline = Date.now() + timeForQuestion * 1000;
    dispatch(setDeadline(newDeadline));
    dispatch(setTimer(timeForQuestion));
  }

  const interval = setInterval(() => {
    if (!deadline) return;

    const remaining = Math.max(
      0,
      Math.floor((deadline - Date.now()) / 1000)
    );

    dispatch(setTimer(remaining));

    if (remaining === 0) {
      clearInterval(interval);
      handleSubmitAnswer();
    }
  }, 1000);

  return () => clearInterval(interval);
}, [currentQuestionIndex, questions, deadline, showWelcomeBack, timer, inProgress]);

  const getTimeByDifficulty = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 20;
      case 'medium': return 60;
      case 'hard': return 120;
      default: return 30;
    }
  };

  const isValidValue = (value: string | null | undefined): boolean => {
    return value !== null && value !== undefined && value.trim() !== '';
  };

  const isInterviewReady = isValidValue(candidateInfo.name) && questions.length > 0;

  const handleFileUpload = async () => {
    if (!file) return;
    setLoading(true);
    try {
      // 1. Parse resume
      const formData = new FormData();
      formData.append('file', file);
      const parseRes = await fetch('https://docura-parser.onrender.com/parse?remove_emails=false', {
        method: 'POST',
        body: formData
      });
      const parseData = await parseRes.json();
      const resumeText = parseData.text;

      // 2. Extract info
      const infoRes = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: 'extract_contact_info', text: resumeText })
      });
      const infoData = await infoRes.json();
      const infoDataText = infoData.text || "";
      const cleanedText = infoDataText.replace(/```(?:json)?\n?/g, '').replace(/```$/g, '').trim();
      
      let parsedInfo: CandidateInfo = {};
      try {
        parsedInfo = JSON.parse(cleanedText);
      } catch (err) {
        console.error("Failed to parse candidate info:", err);
      }

      dispatch(setCandidateInfo(parsedInfo));
      setResumeParsed(true);

      // 3. Check for missing fields
      const fields: string[] = [];
      if (!isValidValue(parsedInfo.name)) fields.push('name');
      if (!isValidValue(parsedInfo.email)) fields.push('email');
      if (!isValidValue(parsedInfo.phone)) fields.push('phone');

      if (fields.length > 0) {
        setMissingFields(fields);
        return;
      }

      // 4. Generate questions
      const qRes = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          task: 'generate_interview_questions', 
          role: 'Full Stack Developer', 
          difficulty: ['easy','medium','hard'], 
          count_per_level: 2 
        })
      });
      const qData = await qRes.json();
      dispatch(setQuestions(qData));
      dispatch(setInProgress(true));
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitMissingInfo = async () => {
    const fields: string[] = [];
    if (!isValidValue(candidateInfo.name)) fields.push('name');
    if (!isValidValue(candidateInfo.email)) fields.push('email');
    if (!isValidValue(candidateInfo.phone)) fields.push('phone');

    if (fields.length > 0) {
      setMissingFields(fields);
      return;
    }

    setMissingFields([]);
    setLoading(true);

    try {
      const qRes = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task: 'generate_interview_questions',
          role: 'Full Stack Developer',
          difficulty: ['easy', 'medium', 'hard'],
          count_per_level: 2
        }),
      });
      const qData = await qRes.json();
      dispatch(setQuestions(qData));
      dispatch(setInProgress(true));
    } catch (err) {
      console.error("Failed to generate questions:", err);
    } finally {
      setLoading(false);
    }
  };

const handleSubmitAnswer = () => {
  if (!questions.length || currentQuestionIndex >= questions.length) return;

  dispatch(setAnswer({ index: currentQuestionIndex, answer: answerInput }));
  setAnswerInput('');
  
  const nextIndex = currentQuestionIndex + 1;
  dispatch(setCurrentQuestionIndex(nextIndex));

  if (nextIndex < questions.length) {
    const nextTime = getTimeByDifficulty(questions[nextIndex].difficulty);
    const newDeadline = Date.now() + nextTime * 1000;
    dispatch(setDeadline(newDeadline));
    dispatch(setTimer(nextTime));
  }
};


const handleContinueInterview = () => {
  setShowWelcomeBack(false);

  // Recalculate deadline with whatever time was left
  const currentQuestion = questions[currentQuestionIndex];
  const remaining = timer || getTimeByDifficulty(currentQuestion.difficulty);
  const newDeadline = Date.now() + remaining * 1000;

  dispatch(setDeadline(newDeadline));
  dispatch(setTimer(remaining));

  setAnswerInput(answers[currentQuestionIndex] || '');
};


  const handleStartFresh = () => {
    dispatch(resetInterview());
    setShowWelcomeBack(false);
    setFile(null);
    setMissingFields([]);
    setResumeParsed(false);
    setAnswerInput('');
    setHasCompletedScoring(false);
  };

  const handleUpdateCandidateInfo = (field: keyof CandidateInfo, value: string) => {
    dispatch(setCandidateInfo({ ...candidateInfo, [field]: value }));
  };

  // Score all answers when interview is completed
useEffect(() => {
  if (
    questions.length &&
    answers.length === questions.length &&
    answers.every(a => a !== undefined) &&
    !hasCompletedScoring
  ) {
    setHasCompletedScoring(true);

    const sendAllScores = async () => {
      try {
        const payload = questions.map((q, idx) => ({
          question: q.text,
          answer: answers[idx] || "",
        }));

        const res = await fetch('/api/gemini', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task: 'score_answer', answers: payload }),
        });

        const scoreData: ScoreData = await res.json();
        console.log("Raw Generated Text:", scoreData);

        const finalScore = scoreData.results.reduce((sum, r) => sum + r.score, 0);

        const chat = scoreData.results.map(r => ({
          question: r.question,
          answer: r.answer,
          score: r.score,
        }));

        const finalSummary = scoreData.summary;

        dispatch(
          addCandidate({
            id: Date.now().toString(),
            name: candidateInfo.name || '',
            email: candidateInfo.email || '',
            phone: candidateInfo.phone || '',
            finalScore,
            finalSummary,
            chat,
          })
        );

        dispatch(markCompleted());
        console.log("Candidate added to Redux");
      } catch (err) {
        console.error("Failed to score answers:", err);
      }
    };

    sendAllScores();
  }
}, [answers.length, questions.length, hasCompletedScoring]);

  // Welcome Back Modal
  if (showWelcomeBack) {
    return (
      <Dialog open={showWelcomeBack} onOpenChange={setShowWelcomeBack}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Welcome Back!</DialogTitle>
            <DialogDescription>
              You have an interview in progress. Would you like to continue where you left off?
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <p className="text-sm text-gray-600">
              Progress: Question {currentQuestionIndex + 1} of {questions.length}
            </p>
            <p className="text-sm text-gray-600 mt-2">
              Candidate: {candidateInfo.name}
            </p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={handleStartFresh}>
              Start Fresh
            </Button>
            <Button onClick={handleContinueInterview}>
              Continue Interview
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  // Step 1: Upload resume
  if (!resumeParsed && !candidateInfo.name && !candidateInfo.email && !candidateInfo.phone) {
    return (
<Card className="max-w-md mx-auto mt-10 bg-gradient-to-br from-pink-50 via-purple-50 to-blue-50 rounded-2xl shadow-lg p-6 border border-gray-100">
  <CardHeader>
    <CardTitle className="text-2xl font-semibold text-gray-800">
      Upload Your Resume
    </CardTitle>
  </CardHeader>
  <CardContent className="flex flex-col gap-4">
    <input 
      type="file" 
      accept=".pdf,.docx" 
      onChange={(e) => setFile(e.target.files?.[0] || null)} 
      className="px-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-200 transition-colors bg-white text-gray-700"
    />
    <Button 
      onClick={handleFileUpload} 
      disabled={!file || loading}
      className={`w-full py-2 rounded-lg font-medium transition-colors 
                  ${loading 
                    ? 'bg-purple-100 text-purple-400 cursor-not-allowed' 
                    : 'bg-purple-200 hover:bg-purple-300 text-purple-800'}`}
    >
      {loading ? 'Processing...' : 'Upload & Start Interview'}
    </Button>
  </CardContent>
</Card>

    );
  }

  // Step 2: Fill missing info
  if (missingFields.length > 0) {
    return (
      <Card className="max-w-md mx-auto mt-10">
        <CardHeader>
          <CardTitle>Complete Your Info</CardTitle>
        </CardHeader>
        <CardContent>
          {missingFields.includes('name') && (
            <Input
              value={candidateInfo.name || ''}
              onChange={(e) => handleUpdateCandidateInfo('name', e.target.value)}
              placeholder="Enter your name"
              className="mb-2"
            />
          )}
          {missingFields.includes('email') && (
            <Input
              value={candidateInfo.email || ''}
              onChange={(e) => handleUpdateCandidateInfo('email', e.target.value)}
              placeholder="Enter your email"
              className="mb-2"
            />
          )}
          {missingFields.includes('phone') && (
            <Input
              value={candidateInfo.phone || ''}
              onChange={(e) => handleUpdateCandidateInfo('phone', e.target.value)}
              placeholder="Enter your phone"
              className="mb-2"
            />
          )}
          <Button onClick={handleSubmitMissingInfo}>
            Continue
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Step 3: Wait for questions
  if (!isInterviewReady) {
    return (
      <Card className="max-w-md mx-auto mt-10">
        <CardHeader>
          <CardTitle>Preparing Interview...</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Please wait while we generate your questions.</p>
        </CardContent>
      </Card>
    );
  }

  // Step 4: Interview completed
  if (currentQuestionIndex >= questions.length) {
    return (
      <Card className="max-w-md mx-auto mt-10">
        <CardHeader>
          <CardTitle>Interview Completed</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Thank you {candidateInfo.name}! Your answers have been submitted.</p>
          <Button onClick={handleStartFresh} className="mt-4">
            Start New Interview
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Step 5: Interview questions
  const currentQuestion = questions[currentQuestionIndex];

  return (
    <Card className="max-w-md mx-auto mt-10">
      <CardHeader>
        <CardTitle>Question {currentQuestionIndex + 1} / {questions.length}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="mb-4">{currentQuestion.text}</p>
        <Input 
          value={answerInput} 
          onChange={(e) => setAnswerInput(e.target.value)} 
          placeholder="Type your answer here..." 
          className="mb-2"
        />
        <Button onClick={handleSubmitAnswer}>Submit Answer</Button>
        <div className="mt-4">
          <p>Time left: {timer}s</p>
          <Progress value={(timer / getTimeByDifficulty(currentQuestion.difficulty)) * 100} />
        </div>
      </CardContent>
    </Card>
  );
};

export default IntervieweeTab;