import { createBrowserClient, createServerClient } from '@supabase/ssr'
import { cookies } from 'next/headers'

export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  )
}

export function createServerSupabaseClient() {
  const cookieStore = cookies()

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return cookieStore.get(name)?.value
        },
        set(name: string, value: string, options: any) {
          cookieStore.set({ name, value, ...options })
        },
        remove(name: string, options: any) {
          cookieStore.set({ name, value: '', ...options })
        },
      },
    }
  )
}

// Database types
export interface Database {
  public: {
    Tables: {
      user_profiles: {
        Row: {
          id: string
          username: string | null
          full_name: string | null
          age_range: string | null
          location_city: string | null
          location_state: string | null
          humor_dimensions: Record<string, number>
          preferences: Record<string, any>
          confidence_score: number | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id: string
          username?: string | null
          full_name?: string | null
          age_range?: string | null
          location_city?: string | null
          location_state?: string | null
          humor_dimensions?: Record<string, number>
          preferences?: Record<string, any>
          confidence_score?: number | null
        }
        Update: {
          username?: string | null
          full_name?: string | null
          age_range?: string | null
          location_city?: string | null
          location_state?: string | null
          humor_dimensions?: Record<string, number>
          preferences?: Record<string, any>
          confidence_score?: number | null
          updated_at?: string
        }
      }
      comedians: {
        Row: {
          id: string
          name: string
          bio: string | null
          humor_vector: Record<string, number>
          image_url: string | null
          spotify_url: string | null
          youtube_url: string | null
          instagram_url: string | null
          popularity_score: number | null
          is_active: boolean | null
          created_at: string
          updated_at: string
        }
      }
      media_items: {
        Row: {
          id: string
          title: string
          type: 'movie' | 'tv_show' | 'special'
          year: number | null
          genres: string[] | null
          humor_attributes: Record<string, number>
          image_url: string | null
          imdb_id: string | null
          created_at: string
        }
      }
      user_interactions: {
        Row: {
          id: string
          user_id: string
          comedian_id: string
          interaction_type: 'like' | 'dislike' | 'save' | 'skip'
          rating: number | null
          created_at: string
        }
        Insert: {
          user_id: string
          comedian_id: string
          interaction_type: 'like' | 'dislike' | 'save' | 'skip'
          rating?: number | null
        }
      }
    }
  }
}