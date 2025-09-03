// frontend/src/app/page.tsx
'use client'
import { useState, useEffect } from 'react'
import { useAuth } from '@/hooks/useAuth'
import { TasteSurvey } from '@/components/TasteSurvey'
import { AuthForm } from '@/components/AuthForm'
import { Dashboard } from '@/components/Dashboard'
import { createClient } from '@/lib/supabase'

export default function HomePage() {
  const { user, loading } = useAuth()
  const [hasProfile, setHasProfile] = useState(false)
  const [profileLoading, setProfileLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const supabase = createClient()

  useEffect(() => {
    if (user) {
      checkProfile()
    } else {
      setProfileLoading(false)
      setHasProfile(false)
    }
  }, [user])

  const checkProfile = async () => {
    try {
      setProfileLoading(true)
      setError(null)
      
      const { data, error: profileError } = await supabase
        .from('user_profiles')
        .select('id')
        .eq('id', user!.id)
        .single()
      
      if (profileError && profileError.code !== 'PGRST116') {
        throw profileError
      }
      
      setHasProfile(!!data)
    } catch (err) {
      console.error('Error checking profile:', err)
      setError('Failed to load profile. Please try again.')
    } finally {
      setProfileLoading(false)
    }
  }

  const handleSurveyComplete = () => {
    setHasProfile(true)
  }

  // Loading state
  if (loading || profileLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md mx-auto p-6">
          <div className="text-red-600 text-xl mb-4">Something went wrong</div>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Refresh Page
          </button>
        </div>
      </div>
    )
  }

  // Not authenticated - show auth form
  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
        <AuthForm />
      </div>
    )
  }

  // Authenticated but no profile - show survey
  if (!hasProfile) {
    return (
      <div className="min-h-screen bg-gray-50 py-8 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">Welcome to Comedy Discovery!</h1>
            <p className="text-xl text-gray-600">Let's learn about your comedy preferences to find your perfect comedians.</p>
          </div>
          <TasteSurvey userId={user.id} onComplete={handleSurveyComplete} />
        </div>
      </div>
    )
  }

  // Authenticated with profile - show dashboard
  return (
    <div className="min-h-screen bg-gray-50">
      <Dashboard userId={user.id} />
    </div>
  )
}