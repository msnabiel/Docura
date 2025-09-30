// store/slices/interviewSlice.ts
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface CandidateInfo {
  name?: string | null;
  email?: string | null;
  phone?: string | null;
}

interface Question {
  text: string;
  difficulty: 'easy' | 'medium' | 'hard';
}

interface InterviewState {
  candidateInfo: CandidateInfo;
  currentQuestionIndex: number;
  answers: string[];
  timer: number;
  questions: Question[];
  deadline: number | null; 
  inProgress: boolean; // for "Welcome Back" modal
}

const initialState: InterviewState = {
  candidateInfo: {},
  currentQuestionIndex: 0,
  answers: [],
  timer: 0,
  deadline: null,
  questions: [],
  inProgress: false,
};

const interviewSlice = createSlice({
  name: 'interview',
  initialState,
  reducers: {
    setCandidateInfo: (state, action: PayloadAction<CandidateInfo>) => {
      state.candidateInfo = action.payload;
    },
    setQuestions: (state, action: PayloadAction<Question[]>) => {
      state.questions = action.payload;
      state.inProgress = true;
    },
    setAnswer: (state, action: PayloadAction<{ index: number; answer: string }>) => {
      state.answers[action.payload.index] = action.payload.answer;
    },
    setCurrentQuestionIndex: (state, action: PayloadAction<number>) => {
      state.currentQuestionIndex = action.payload;
    },
    setTimer: (state, action: PayloadAction<number>) => {
      state.timer = action.payload;
    },
    setInProgress: (state, action: PayloadAction<boolean>) => {   // <-- define reducer
      state.inProgress = action.payload;
    },
    setDeadline: (state, action: PayloadAction<number | null>) => {   // ✅ typed
      state.deadline = action.payload;
    },
    resetInterview: () => initialState,
    markCompleted: (state) => {
      state.inProgress = false;
    },
  },
});

export const {
  setCandidateInfo,
  setQuestions,
  setAnswer,
  setCurrentQuestionIndex,
  setTimer,
  setDeadline,
setInProgress,
  resetInterview,
  markCompleted,
} = interviewSlice.actions;

export default interviewSlice.reducer;
