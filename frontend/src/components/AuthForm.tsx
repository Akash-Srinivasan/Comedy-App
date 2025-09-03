// frontend/src/components/Dashboard.tsx
'use client'
import { useState, useEffect } from 'react'
import { createClient } from '@/lib/supabase'
import { useAuth } from '@/hooks/useAuth'
import { RecommendationEngine } from '@/lib/recommendationEngine'
import type { Database } from '@/lib/supabase'
import { Heart, BookmarkIcon, X, ExternalLink, MapPin, Calendar, Menu, User, LogOut } from 'lucide-react'

interface DashboardProps {
  userId: string
}

type Comedian = Database['public']['Tables']['comedians']['Row']
type UserProfile = Database['public']['Tables']['user_profiles']['Row']

interface RecommendationScore {
  comedian: Comedian
  score: number
  reasons: string[]
}

export function Dashboard({ userId }: DashboardProps) {
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [recommendations, setRecommendations] = useState<RecommendationScore[]>([])
  const [loading, setLoading] = useState(true)
  const [interactionLoading, setInteractionLoading] = useState<string | null>(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [showMenu, setShowMenu] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const { signOut, user } = useAuth()
  const supabase = createClient()
  const recommendationEngine = new RecommendationEngine()

  useEffect(() => {
    loadUserData()
  }, [])

  const loadUserData = async () => {
    try {
      setLoading(true)
      setError(null)

      // Load user profile
      const { data: profileData, error: profileError } = await supabase
        .from('user_profiles')
        .select('*')
        .eq('id', userId)
        .single()

      if (profileError) throw profileError
      setProfile(profileData)

      // Load recommendations
      const recs = await recommendationEngine.getRecommendations(userId, 20)
      setRecommendations(recs)

    } catch (err) {
      console.error('Error loading user data:', err)
      setError('Failed to load recommendations. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleInteraction = async (
    comedianId: string,
    interactionType: 'like' | 'dislike' | 'save' | 'skip',
    rating?: number
  ) => {
    try {
      setInteractionLoading(comedianId)
      
      await recommendationEngine.recordInteraction(userId, comedianId, interactionType, rating)
      
      // Move to next comedian
      if (currentIndex < recommendations.length - 1) {
        setCurrentIndex(currentIndex + 1)
      } else {
        // Load more recommendations when we reach the end
        const newRecs = await recommendationEngine.getRecommendations(userId, 10)
        if (newRecs.length > 0) {
          setRecommendations(prev => [...prev, ...newRecs])
          setCurrentIndex(currentIndex + 1)
        }
      }
    } catch (err) {
      console.error('Error recording interaction:', err)
      setError('Failed to save your preference. Please try again.')
    } finally {
      setInteractionLoading(null)
    }
  }

  const handleSignOut = async () => {
    await signOut()
  }

  const currentComedian = recommendations[currentIndex]

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading your recommendations...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center max-w-md mx-auto p-6">
          <div className="text-red-600 text-xl mb-4">Something went wrong</div>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={loadUserData}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  if (!currentComedian) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center max-w-md mx-auto p-6">
          <div className="text-gray-800 text-xl mb-4">No more recommendations</div>
          <p className="text-gray-600 mb-6">
            You've seen all available comedians! Check back later for new additions.
          </p>
          <button
            onClick={loadUserData}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">Comedy Discovery</h1>
              {profile && (
                <div className="ml-4 text-sm text-gray-600">
                  Welcome, {profile.full_name || user?.email}
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-500">
                {currentIndex + 1} of {recommendations.length}
              </div>
              <div className="relative">
                <button
                  onClick={() => setShowMenu(!showMenu)}
                  className="p-2 text-gray-600 hover:text-gray-900"
                >
                  <Menu className="w-5 h-5" />
                </button>
                
                {showMenu && (
                  <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10">
                    <button
                      onClick={() => {
                        setShowMenu(false)
                        // Add profile editing functionality here
                      }}
                      className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full"
                    >
                      <User className="w-4 h-4 mr-2" />
                      Edit Profile
                    </button>
                    <button
                      onClick={handleSignOut}
                      className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full"
                    >
                      <LogOut className="w-4 h-4 mr-2" />
                      Sign Out
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* Comedian Card */}
          <div className="relative">
            {currentComedian.comedian.image_url ? (
              <img
                src={currentComedian.comedian.image_url}
                alt={currentComedian.comedian.name}
                className="w-full h-96 object-cover"
              />
            ) : (
              <div className="w-full h-96 bg-gradient-to-br from-blue-400 to-purple-600 flex items-center justify-center">
                <div className="text-white text-6xl font-bold">
                  {currentComedian.comedian.name.charAt(0)}
                </div>
              </div>
            )}
            
            {/* Overlay with name and basic info */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-6">
              <h2 className="text-3xl font-bold text-white mb-2">
                {currentComedian.comedian.name}
              </h2>
              <div className="flex items-center space-x-4 text-white/90">
                <div className="flex items-center">
                  <span className="text-sm">Match: {Math.round(currentComedian.score * 100)}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Comedian Details */}
          <div className="p-6">
            {currentComedian.comedian.bio && (
              <p className="text-gray-700 mb-4 leading-relaxed">
                {currentComedian.comedian.bio}
              </p>
            )}

            {/* Why recommended */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Why you might like this</h3>
              <div className="flex flex-wrap gap-2">
                {currentComedian.reasons.map((reason, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                  >
                    {reason}
                  </span>
                ))}
              </div>
            </div>

            {/* Humor dimensions */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Comedy Style</h3>
              <div className="space-y-2">
                {Object.entries(currentComedian.comedian.humor_vector as Record<string, number>)
                  .filter(([_, value]) => value > 0.3)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 5)
                  .map(([dimension, value]) => (
                    <div key={dimension} className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 capitalize">
                        {dimension.replace('_', ' ')}
                      </span>
                      <div className="flex-1 mx-3 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full"
                          style={{ width: `${value * 100}%` }}
                        />
                      </div>
                      <span className="text-sm text-gray-500 w-8">
                        {Math.round(value * 100)}%
                      </span>
                    </div>
                  ))}
              </div>
            </div>

            {/* Social Links */}
            {(currentComedian.comedian.spotify_url || currentComedian.comedian.youtube_url || currentComedian.comedian.instagram_url) && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Find them online</h3>
                <div className="flex space-x-3">
                  {currentComedian.comedian.spotify_url && (
                    <a
                      href={currentComedian.comedian.spotify_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Spotify
                    </a>
                  )}
                  {currentComedian.comedian.youtube_url && (
                    <a
                      href={currentComedian.comedian.youtube_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      YouTube
                    </a>
                  )}
                  {currentComedian.comedian.instagram_url && (
                    <a
                      href={currentComedian.comedian.instagram_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Instagram
                    </a>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="px-6 pb-6">
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => handleInteraction(currentComedian.comedian.id, 'dislike')}
                disabled={interactionLoading === currentComedian.comedian.id}
                className="flex-1 max-w-24 bg-red-100 text-red-700 p-4 rounded-xl hover:bg-red-200 transition-colors disabled:opacity-50"
              >
                <X className="w-6 h-6 mx-auto" />
              </button>
              
              <button
                onClick={() => handleInteraction(currentComedian.comedian.id, 'save')}
                disabled={interactionLoading === currentComedian.comedian.id}
                className="flex-1 max-w-24 bg-blue-100 text-blue-700 p-4 rounded-xl hover:bg-blue-200 transition-colors disabled:opacity-50"
              >
                <BookmarkIcon className="w-6 h-6 mx-auto" />
              </button>
              
              <button
                onClick={() => handleInteraction(currentComedian.comedian.id, 'like', 4)}
                disabled={interactionLoading === currentComedian.comedian.id}
                className="flex-1 max-w-24 bg-green-100 text-green-700 p-4 rounded-xl hover:bg-green-200 transition-colors disabled:opacity-50"
              >
                <Heart className="w-6 h-6 mx-auto" />
              </button>
            </div>
            
            <div className="flex justify-between text-xs text-gray-500 mt-2 px-4">
              <span>Pass</span>
              <span>Save</span>
              <span>Like</span>
            </div>
          </div>
        </div>

        {/* Progress indicator */}
        <div className="mt-6 bg-white rounded-lg p-4 shadow">
          <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
            <span>Discovery Progress</span>
            <span>{currentIndex + 1} / {recommendations.length}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${((currentIndex + 1) / recommendations.length) * 100}%` }}
            />
          </div>
        </div>
      </main>
    </div>
  )
}