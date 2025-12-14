// frontend/src/lib/api.ts
// Define TypeScript types to match the API contracts
export interface ChatRequest {
  message: string;
  context?: string;
  sessionId?: string;
  userId?: string;
}

export interface ChatResponse {
  id: string;
  response: string;
  timestamp: string;
  contextUsed: boolean;
  sources?: string[];
}

export interface PersonalizeRequest {
  chapterId: string;
  userId: string;
  learningStyle?: 'visual' | 'auditory' | 'reading' | 'kinesthetic';
  preferences?: Record<string, any>;
}

export interface PersonalizeResponse {
  chapterId: string;
  personalizedContent: string;
  metadata: {
    processingTime: number;
  };
}

export interface TranslateRequest {
  chapterId: string;
  userId: string;
  targetLanguage: string;
  preserveFormat?: boolean;
}

export interface TranslateResponse {
  chapterId: string;
  translatedContent: string;
  targetLanguage: string;
  metadata: {
    processingTime: number;
  };
}

export interface ChapterProgressResponse {
  userId: string;
  chapterId: string;
  completionPercentage: number;
  timeSpent: number;
  lastAccessed: string;
  bookmarks: number[];
}

export interface UpdateProgressRequest {
  completionPercentage: number;
  timeSpent: number;
  bookmarks?: number[];
}

const API_BASE_URL = typeof window !== 'undefined'
  ? (window as any).API_BASE_URL || 'http://localhost:8000'
  : 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `API request failed with status ${response.status}`);
    }

    return response.json();
  }

  async personalize(request: PersonalizeRequest): Promise<PersonalizeResponse> {
    const response = await fetch(`${this.baseUrl}/api/personalize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `API request failed with status ${response.status}`);
    }

    return response.json();
  }

  async translate(request: TranslateRequest): Promise<TranslateResponse> {
    const response = await fetch(`${this.baseUrl}/api/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `API request failed with status ${response.status}`);
    }

    return response.json();
  }

  async getChapterProgress(chapterId: string, userId?: string): Promise<ChapterProgressResponse> {
    let url = `${this.baseUrl}/api/progress/${chapterId}`;
    if (userId) {
      url += `?userId=${encodeURIComponent(userId)}`;
    }

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `API request failed with status ${response.status}`);
    }

    return response.json();
  }

  async updateChapterProgress(chapterId: string, request: UpdateProgressRequest, userId?: string): Promise<ChapterProgressResponse> {
    let url = `${this.baseUrl}/api/progress/${chapterId}`;
    if (userId) {
      url += `?userId=${encodeURIComponent(userId)}`;
    }

    const response = await fetch(url, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `API request failed with status ${response.status}`);
    }

    return response.json();
  }
}

export const apiClient = new ApiClient();