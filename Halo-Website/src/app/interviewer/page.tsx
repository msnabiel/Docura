"use client";
import React, { useState, useMemo } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useSelector, useDispatch } from "react-redux";
import { RootState } from "@/store";
import { deleteCandidate } from "@/store/slices/candidateSlice"; // Import the delete action

// Types
interface CandidateSummary {
  id: string;
  name: string;
  email: string;
  phone: string;
  finalScore: number;
  finalSummary: string;
  chat: { question: string; answer: string; score: number }[];
}

export default function InterviewerDashboard() {
  const dispatch = useDispatch(); // Initialize the dispatch function
  const candidates = useSelector(
    (state: RootState) => state.candidates.candidates
  );
    const handleDeleteCandidate = (id: string) => {
  dispatch(deleteCandidate(id));
};
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<CandidateSummary | null>(null);

  const filtered = useMemo(() => {
    return candidates
      .filter((c) =>
        c.name.toLowerCase().includes(search.toLowerCase())
      )
      .sort((a, b) => b.finalScore - a.finalScore);
  }, [search, candidates]);

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      <main className="p-6 space-y-6 max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900">Interviewer Dashboard</h1>

        <div className="flex flex-col sm:flex-row gap-4 sm:items-center">
          <Input
            placeholder="Search candidates by name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 border-gray-300 bg-white shadow-sm focus:ring-2 focus:ring-blue-200 rounded-md"
          />
        </div>

        <div className="overflow-x-auto bg-white rounded-xl shadow-md">
          <Table className="min-w-full">
            <TableHeader className="bg-gray-100">
              <TableRow>
                <TableHead className="text-left text-gray-600">Name</TableHead>
                <TableHead className="text-left text-gray-600">Email</TableHead>
                <TableHead className="text-left text-gray-600">Phone</TableHead>
                <TableHead className="text-left text-gray-600">Score</TableHead>
                <TableHead className="text-left text-gray-600">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((c) => (
                <TableRow
                  key={c.id}
                  className="hover:bg-gray-50 transition-colors"
                >
                  <TableCell className="py-2">{c.name}</TableCell>
                  <TableCell className="py-2">{c.email}</TableCell>
                  <TableCell className="py-2">{c.phone}</TableCell>
                  <TableCell className="py-2 font-medium">{c.finalScore}</TableCell>
                  <TableCell className="py-2 flex gap-2">
                    <Button
                      variant="outline"
                      className="bg-white text-blue-600 border-blue-200 hover:bg-blue-50"
                      onClick={() => {
                        const updatedCandidate = candidates.find((x) => x.id === c.id);
                        setSelected(updatedCandidate || c);
                      }}
                    >
                      View
                    </Button>

                    <Button
                      variant="destructive"
                      className="bg-red-600 text-white hover:bg-red-500 p-2"
                      onClick={() => {
                        if (
                          confirm(`Are you sure you want to delete candidate "${c.name}"?`)
                        ) {
                          handleDeleteCandidate(c.id) // Dispatch the delete action
                        }
                      }}
                    >
                      <Trash2 size={18} />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        <Dialog open={!!selected} onOpenChange={() => setSelected(null)}>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto rounded-xl p-6 bg-white shadow-lg">
            <DialogHeader>
              <DialogTitle className="text-xl font-semibold text-gray-900">
                Candidate Details
              </DialogTitle>
            </DialogHeader>

            {selected && (
              <div className="space-y-6">
                <div className="space-y-1">
                  <p><strong>Name:</strong> {selected.name}</p>
                  <p><strong>Email:</strong> {selected.email}</p>
                  <p><strong>Phone:</strong> {selected.phone}</p>
                  <p><strong>Final Score:</strong> {selected.finalScore}</p>
                  <p><strong>Summary:</strong> {selected.finalSummary}</p>
                </div>

                <div>
                  <h3 className="font-semibold mb-2 text-gray-800">Q&A Transcript</h3>
                  <div className="space-y-3">
                    {selected.chat?.map((c, i) => (
                      <div
                        key={i}
                        className="p-3 rounded-lg border border-gray-200 bg-gray-50 shadow-sm"
                      >
                        <p className="font-medium text-gray-700">Q{i + 1}: {c.question}</p>
                        <p className="text-gray-600"><strong>Answer:</strong> {c.answer}</p>
                        <p className="text-gray-600"><strong>Score:</strong> {c.score}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </main>
    </div>
  );
}
