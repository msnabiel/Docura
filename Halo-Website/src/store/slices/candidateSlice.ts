import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface Candidate {
  id: string;
  name: string;
  email: string;
  phone: string;
  finalScore: number;
  finalSummary: string;
  chat: { question: string; answer: string; score: number }[];
}

interface CandidateState {
  candidates: Candidate[];
}

const initialState: CandidateState = {
  candidates: [],
};

const candidateSlice = createSlice({
  name: 'candidates',
  initialState,
  reducers: {
    addCandidate: (state, action: PayloadAction<Candidate>) => {
      state.candidates.push(action.payload);
    },
    deleteCandidate: (state, action: PayloadAction<string>) => {
      state.candidates = state.candidates.filter(
        (candidate) => candidate.id !== action.payload
      );
    },
  },
});

export const { addCandidate, deleteCandidate } = candidateSlice.actions;
export default candidateSlice.reducer;
