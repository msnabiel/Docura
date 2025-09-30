// app/api/gemini/route.ts
import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";

const GOOGLE_KEY = process.env.GOOGLE_KEY || "YOUR_API_KEY";

const ai = new GoogleGenAI({
  apiKey: GOOGLE_KEY,
});
async function retry<T>(
  fn: () => Promise<T>,
  retries = 3,
  delayMs = 1000
): Promise<T> {
  let lastError: unknown;
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      console.warn(`Attempt ${i + 1} failed. Retrying in ${delayMs}ms...`);
      await new Promise((res) => setTimeout(res, delayMs));
    }
  }
  // Optionally narrow type if you want to throw as Error
  if (lastError instanceof Error) throw lastError;
  throw new Error("Unknown error occurred during retry");
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { task, text, role, difficulty, count_per_level, question, answer, answers } = body;

    let prompt = "";

    switch (task) {
      case "extract_contact_info":
        prompt = `Extract name, email, and phone number from the following resume and return in JSON with keys name, email, phone. If any is missing set it to null.\n\n${text}`;
        break;

      case "generate_interview_questions":
        prompt = `Generate interview questions for role: ${role}.\nDifficulty levels: ${difficulty.join(
          ", "
        )}\nGenerate ${count_per_level} questions per difficulty level. Return as JSON array with text and difficulty.`;
        break;

        case "score_answer":
            if (!Array.isArray(answers) || answers.length === 0) {
                return NextResponse.json({ error: "Provide an array of question-answer pairs in 'answers'" }, { status: 400 });
            }

            // Construct one prompt for all Q&A and ask for summary
            prompt = `Score the following question-answer pairs on a scale of 0-10 and return JSON with two keys:
            1. "results": array of objects {question, answer, score}
            2. "summary": a short feedback/overall assessment based on all answers

            Here are the question-answer pairs:\n\n`;

            answers.forEach((qa: { question: string; answer: string }) => {
                prompt += `Question: ${qa.question}\nAnswer: ${qa.answer}\n\n`;
            });
            break;

      default:
        return NextResponse.json({ error: "Unknown task" }, { status: 400 });
    }

  const response = await retry(() =>
    ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
    }),
    3,
    1000
  );

    // Parsing JSON if needed
    if (task === "generate_interview_questions" || task === "score_answer") {
      const text = response.text ?? "";
      const cleanedText = text.replace(/```(?:json)?\n?/g, "").replace(/```$/g, "").trim();
      console.log("Raw Generated Text:", cleanedText);
      try {
        const parsed = JSON.parse(cleanedText);
        return NextResponse.json(parsed);
      } catch (err) {
        return NextResponse.json({ raw: text });
      }
    }

    return NextResponse.json({ text: response.text });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
  }
}
